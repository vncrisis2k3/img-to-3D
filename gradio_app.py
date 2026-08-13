# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import random
import shutil
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid

from hy3dgen.shapegen.utils import logger

MAX_SEED = int(1e7)


def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def get_example_mv_list():
    print('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
    for mv_dir in os.listdir(root):
        view_list = []
        for view in ['front', 'back', 'left', 'right']:
            path = os.path.join(root, mv_dir, f'{view}.png')
            if os.path.exists(path):
                view_list.append(path)
            else:
                view_list.append(None)
        mv_list.append(view_list)
    return mv_list


def gen_save_folder(max_size=200):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 获取所有文件夹路径
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]

    # 如果文件夹数量超过 max_size，删除创建时间最久的文件夹
    if len(dirs) >= max_size:
        # 按创建时间排序，最久的排在前面
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")

    # 生成一个新的 uuid 文件夹名称
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")

    return new_folder


def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    progress=None,
):
    if progress:
        progress(0.02, desc="Đang kiểm tra đầu vào")
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Vui lòng cung cấp ảnh hoặc mô tả văn bản.")
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Vui lòng cung cấp ít nhất một ảnh góc nhìn.")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right

    seed = int(randomize_seed_fn(seed, randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}

    if image is None:
        start_time = time.time()
        if progress:
            progress(0.08, desc="Đang tạo ảnh từ mô tả")
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error("Text to 3D đang tắt. Hãy chạy `python gradio_app.py --enable_t23d` để bật.")
        time_meta['text2image'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'input.png'))
    if MV_MODE:
        start_time = time.time()
        if progress:
            progress(0.15, desc="Đang xử lý nền ảnh nhiều góc")
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = rmbg_worker(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            if progress:
                progress(0.15, desc="Đang xóa nền ảnh")
            image = rmbg_worker(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()
    if progress:
        progress(0.28, desc="Đang sinh mesh 3D")

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    if progress:
        progress(0.78, desc="Đang chuyển đổi sang mesh")
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    if progress:
        progress(0.86, desc="Đã tạo mesh")
    return mesh, main_image, save_folder, stats, seed


def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    start_time_0 = time.time()
    progress(0.01, desc="Đang bắt đầu tạo mesh có texture")
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
        progress=progress,
    )
    progress(0.88, desc="Đang lưu mesh trắng")
    path = export_mesh(mesh, save_folder, textured=False)

    # tmp_time = time.time()
    # mesh = floater_remove_worker(mesh)
    # mesh = degenerate_face_remove_worker(mesh)
    # logger.info("---Postprocessing takes %s seconds ---" % (time.time() - tmp_time))
    # stats['time']['postprocessing'] = time.time() - tmp_time

    tmp_time = time.time()
    progress(0.90, desc="Đang tối ưu số mặt")
    mesh = face_reduce_worker(mesh)
    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

    tmp_time = time.time()
    progress(0.94, desc="Đang tạo texture")
    textured_mesh = texgen_worker(mesh, image)
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0

    textured_mesh.metadata['extras'] = stats
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    progress(0.98, desc="Đang dựng khung xem trước")
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                         textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        gr.update(value=path_textured),
        model_viewer_html_textured,
        stats,
        seed,
    )


def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    start_time_0 = time.time()
    progress(0.01, desc="Đang bắt đầu tạo mesh")
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
        progress=progress,
    )
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats

    progress(0.92, desc="Đang lưu mesh")
    path = export_mesh(mesh, save_folder, textured=False)
    progress(0.97, desc="Đang dựng khung xem trước")
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    mode_badge = "Bản Mini Turbo" if TURBO_MODE and 'mini' in args.subfolder else "Hunyuan3D"
    texture_badge = "Có tạo texture" if HAS_TEXTUREGEN else "Chỉ tạo mesh"
    title_html = f"""
    <section class="app-hero">
      <div>
        <p class="app-kicker">AI Ảnh sang 3D</p>
        <h1>Tạo mô hình 3D từ ảnh</h1>
        <p class="app-subtitle">Biến ảnh đầu vào thành mesh 3D, tạo texture khi khả dụng và xuất file GLB, OBJ, PLY hoặc STL.</p>
        <div class="app-badges">
          <span>{mode_badge}</span>
          <span>{texture_badge}</span>
          <span>GPU Ready</span>
        </div>
      </div>
      <nav class="app-links">
        <a href="https://github.com/tencent/Hunyuan3D-2">GitHub</a>
        <a href="http://3d-models.hunyuan.tencent.com">Trang chủ</a>
        <a href="https://3d.hunyuan.tencent.com">Studio</a>
        <a href="#">Báo cáo</a>
        <a href="https://huggingface.co/Tencent/Hunyuan3D-2">Model</a>
      </nav>
      <div id="connection-toast" class="connection-toast" aria-live="polite">
        <strong>Đang kết nối lại</strong>
        <span>Hệ thống sẽ tự thử lại trong giây lát.</span>
      </div>
      <script>
      (() => {{
        const toast = document.getElementById("connection-toast");
        if (!toast) return;
        let retryDelay = 2000;
        const maxDelay = 15000;
        const show = (title, detail, mode) => {{
          toast.querySelector("strong").textContent = title;
          toast.querySelector("span").textContent = detail;
          toast.dataset.mode = mode;
          toast.classList.add("visible");
        }};
        const hide = () => {{
          toast.classList.remove("visible");
          retryDelay = 2000;
        }};
        const ping = async () => {{
          try {{
            const response = await fetch("/health", {{ cache: "no-store" }});
            if (!response.ok) throw new Error("health check failed");
            hide();
            setTimeout(ping, 5000);
          }} catch (error) {{
            show("Đang kết nối lại", `Thử lại sau ${{Math.round(retryDelay / 1000)}} giây...`, "warning");
            setTimeout(ping, retryDelay);
            retryDelay = Math.min(maxDelay, Math.round(retryDelay * 1.6));
          }}
        }};
        window.addEventListener("offline", () => show("Mất mạng", "Kiểm tra kết nối rồi hệ thống sẽ tự thử lại.", "error"));
        window.addEventListener("online", () => {{
          show("Đã có mạng", "Đang nối lại server...", "success");
          retryDelay = 1000;
          ping();
        }});
        const placeholderSvg = encodeURIComponent(`
          <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">
            <rect width="96" height="96" rx="10" fill="#e9f0f7"/>
            <path d="M25 62l16-18 12 13 8-9 12 14H25z" fill="#9fb3ca"/>
            <circle cx="63" cy="32" r="7" fill="#b8c7df"/>
          </svg>
        `);
        document.addEventListener("error", (event) => {{
          const target = event.target;
          if (target && target.tagName === "IMG" && target.closest(".studio-gallery")) {{
            target.classList.add("img-fallback");
            target.src = `data:image/svg+xml;charset=utf-8,${{placeholderSvg}}`;
          }}
        }}, true);
        setTimeout(ping, 3000);
      }})();
      </script>
    </section>
    """
    custom_css = """
    :root {
        --studio-bg: #eef2f7;
        --studio-panel: #ffffff;
        --studio-border: #cfd9e8;
        --studio-text: #111827;
        --studio-muted: #5b6677;
        --studio-accent: #0f766e;
        --studio-accent-strong: #0d5f59;
        --studio-blue: #2563eb;
        --studio-soft: #e9f7f5;
        --studio-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
    }

    body, gradio-app {
        background: var(--studio-bg) !important;
        color: var(--studio-text);
    }

    .gradio-container {
        max-width: 1680px !important;
        padding: 18px 22px 24px !important;
        background: var(--studio-bg) !important;
    }

    #component-0,
    .contain {
        background: var(--studio-bg) !important;
    }

    .app-hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        padding: 24px 28px;
        margin-bottom: 18px;
        border: 1px solid var(--studio-border);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(15, 118, 110, 0.10), rgba(37, 99, 235, 0.08)),
            #ffffff;
        box-shadow: var(--studio-shadow);
    }

    .app-hero h1 {
        margin: 4px 0 8px;
        font-size: clamp(28px, 3vw, 44px);
        line-height: 1.05;
        letter-spacing: 0;
        color: #0f172a;
    }

    .app-kicker {
        margin: 0;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--studio-accent);
    }

    .app-subtitle {
        margin: 0;
        max-width: 760px;
        color: var(--studio-muted);
        font-size: 15px;
        line-height: 1.5;
    }

    .app-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }

    .app-badges span {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 0 10px;
        border: 1px solid rgba(15, 118, 110, 0.22);
        border-radius: 8px;
        background: #ffffff;
        color: #0f4f4a;
        font-size: 12px;
        font-weight: 700;
    }

    .app-links {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
        min-width: 280px;
    }

    .app-links a {
        display: inline-flex;
        align-items: center;
        min-height: 32px;
        padding: 0 12px;
        border: 1px solid #cbd8ee;
        border-radius: 8px;
        background: #ffffff;
        color: #1f3b71;
        font-size: 13px;
        font-weight: 650;
        text-decoration: none;
    }

    .app-links a:hover {
        border-color: var(--studio-accent);
        color: var(--studio-accent-strong);
    }

    .studio-grid {
        gap: 16px !important;
        align-items: stretch !important;
    }

    .studio-panel {
        min-width: 0 !important;
        padding: 16px !important;
        border: 1px solid var(--studio-border) !important;
        border-radius: 8px !important;
        background: var(--studio-panel) !important;
        box-shadow: var(--studio-shadow);
    }

    .studio-panel > div,
    .studio-viewer > div,
    .studio-gallery > div {
        background: transparent !important;
    }

    .studio-viewer {
        padding: 16px !important;
        border: 1px solid var(--studio-border) !important;
        border-radius: 8px !important;
        background: #fbfdff !important;
        box-shadow: var(--studio-shadow);
    }

    .studio-gallery {
        min-width: 280px !important;
        padding: 16px !important;
        border: 1px solid var(--studio-border) !important;
        border-radius: 8px !important;
        background: var(--studio-panel) !important;
        box-shadow: var(--studio-shadow);
    }

    .studio-panel .tabs,
    .studio-viewer .tabs,
    .studio-gallery .tabs {
        border: 0 !important;
        background: transparent !important;
    }

    .studio-panel .wrap,
    .studio-viewer .wrap,
    .studio-gallery .wrap {
        border-radius: 8px !important;
    }

    .tab-nav {
        gap: 6px !important;
        border-bottom: 1px solid var(--studio-border) !important;
    }

    .tab-nav button {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 650 !important;
        color: #4b5563 !important;
    }

    .tab-nav button.selected {
        color: var(--studio-accent-strong) !important;
        border-color: var(--studio-accent) !important;
        background: var(--studio-soft) !important;
    }

    .tabs > .tabitem {
        padding-top: 12px !important;
    }

    .primary-action button,
    button.primary {
        border-radius: 8px !important;
        background: var(--studio-accent) !important;
        border-color: var(--studio-accent) !important;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.24) !important;
        font-weight: 700 !important;
    }

    .primary-action button:hover,
    button.primary:hover {
        background: var(--studio-accent-strong) !important;
    }

    .primary-action button:disabled,
    .secondary-action button:disabled,
    button:disabled {
        opacity: 0.62 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    .progress-text,
    .progress-bar {
        color: var(--studio-accent-strong) !important;
    }

    .secondary-action button {
        border-radius: 8px !important;
        font-weight: 650 !important;
    }

    .panel-heading {
        margin: 0 0 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--studio-border);
    }

    .panel-heading strong {
        display: block;
        color: #0f172a;
        font-size: 16px;
        line-height: 1.2;
    }

    .panel-heading span {
        display: block;
        margin-top: 4px;
        color: var(--studio-muted);
        font-size: 12px;
        line-height: 1.4;
    }

    .studio-status {
        margin-top: 14px;
        padding: 12px 16px;
        border: 1px solid var(--studio-border);
        border-radius: 8px;
        background: #ffffff;
        color: var(--studio-muted);
        text-align: center;
        font-size: 13px;
    }

    .studio-warning {
        margin-top: 8px;
        padding: 10px 14px;
        border: 1px solid #f6c768;
        border-radius: 8px;
        background: #fff8e6;
        color: #7c5600;
        text-align: center;
        font-size: 13px;
    }

    .connection-toast {
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        display: grid;
        gap: 3px;
        width: min(320px, calc(100vw - 36px));
        padding: 12px 14px;
        border: 1px solid #f6c768;
        border-radius: 8px;
        background: #fff8e6;
        color: #7c5600;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.16);
        opacity: 0;
        pointer-events: none;
        transform: translateY(-8px);
        transition: opacity 160ms ease, transform 160ms ease;
    }

    .connection-toast.visible {
        opacity: 1;
        transform: translateY(0);
    }

    .connection-toast strong {
        font-size: 13px;
        line-height: 1.2;
    }

    .connection-toast span {
        font-size: 12px;
        line-height: 1.4;
    }

    .connection-toast[data-mode="error"] {
        border-color: #fecaca;
        background: #fff1f2;
        color: #9f1239;
    }

    .connection-toast[data-mode="success"] {
        border-color: #bbf7d0;
        background: #f0fdf4;
        color: #166534;
    }

    .toast-wrap,
    .toast-container {
        top: 18px !important;
        right: 18px !important;
        left: auto !important;
        width: min(340px, calc(100vw - 36px)) !important;
        pointer-events: none !important;
    }

    .toast-wrap > *,
    .toast-container > * {
        border-radius: 8px !important;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.14) !important;
        pointer-events: auto !important;
    }

    .studio-gallery .gallery,
    .studio-gallery table {
        width: 100% !important;
    }

    .studio-gallery img {
        width: 64px !important;
        height: 64px !important;
        object-fit: cover !important;
        border-radius: 8px !important;
        border: 1px solid var(--studio-border) !important;
        background:
            linear-gradient(90deg, #eef3f8 25%, #f8fbff 37%, #eef3f8 63%) !important;
        background-size: 400% 100% !important;
        animation: skeleton-loading 1.4s ease infinite !important;
    }

    .studio-gallery img:not(.img-fallback)[src] {
        animation: none !important;
        background: #f8fafc !important;
    }

    .studio-gallery td,
    .studio-gallery th {
        padding: 4px !important;
        border-color: transparent !important;
    }

    @keyframes skeleton-loading {
        0% { background-position: 100% 50%; }
        100% { background-position: 0 50%; }
    }

    .studio-panel [data-testid="image"],
    .studio-panel .image-container,
    .studio-panel .dropzone {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .studio-panel .dropzone {
        border: 1px dashed #9fb3ca !important;
        background: #f8fbff !important;
    }

    .studio-panel textarea,
    .studio-panel input,
    .studio-panel select {
        border-radius: 8px !important;
    }

    .studio-panel label,
    .studio-viewer label,
    .studio-gallery label {
        color: #253043 !important;
        font-weight: 650 !important;
    }

    .studio-panel .form,
    .studio-panel .block,
    .studio-viewer .block,
    .studio-gallery .block {
        border-color: #dbe3ef !important;
        border-radius: 8px !important;
    }

    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    .empty-viewer {
        min-height: 650px;
        width: 100%;
        border-radius: 8px;
        border: 1px dashed #b8c7df;
        background:
            linear-gradient(135deg, rgba(15, 118, 110, 0.10), rgba(37, 99, 235, 0.10)),
            #f8fbff;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .empty-viewer div {
        text-align: center;
        color: #5b6677;
    }

    .empty-viewer strong {
        display: block;
        color: #111827;
        font-size: 18px;
        margin-bottom: 6px;
    }

    @media (max-width: 1280px) {
        .studio-grid {
            flex-wrap: wrap !important;
        }

        .studio-panel {
            flex: 1 1 360px !important;
        }

        .studio-viewer {
            flex: 2 1 620px !important;
        }

        .studio-gallery {
            flex: 1 1 100% !important;
        }
    }

    @media (max-width: 1100px) {
        .app-hero {
            align-items: flex-start;
            flex-direction: column;
        }

        .app-links {
            justify-content: flex-start;
            min-width: 0;
        }

        .empty-viewer {
            min-height: 460px;
        }
    }

    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(f"<style>{custom_css}</style>{title_html}")

        with gr.Row(elem_classes='studio-grid'):
            with gr.Column(scale=3, elem_classes='studio-panel'):
                gr.HTML("""
                <div class="panel-heading">
                    <strong>Nguồn đầu vào</strong>
                    <span>Tải ảnh rõ chủ thể hoặc chọn một mẫu có sẵn.</span>
                </div>
                """)
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Ảnh đầu vào', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='Ảnh nguồn', type='pil', image_mode='RGBA', height=330)

                    with gr.Tab('Mô tả văn bản', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
                        caption = gr.Textbox(label='Mô tả',
                                             placeholder='Ví dụ: mô hình 3D một chú mèo trắng, nền trắng.',
                                             info='HunyuanDiT sẽ tạo ảnh trung gian từ mô tả này.')
                    with gr.Tab('Nhiều góc nhìn', visible=MV_MODE) as tab_mv:
                        # gr.Label('Please upload at least one front image.')
                        with gr.Row():
                            mv_image_front = gr.Image(label='Mặt trước', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Mặt sau', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Bên trái', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Bên phải', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row(equal_height=True):
                    btn = gr.Button(value='Tạo mesh', variant='primary', min_width=100, elem_classes='primary-action')
                    btn_all = gr.Button(value='Tạo mesh có texture',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100,
                                        elem_classes='primary-action')

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                gr.HTML("""
                <div class="panel-heading" style="margin-top: 14px;">
                    <strong>Thiết lập</strong>
                    <span>Tinh chỉnh tốc độ, chất lượng mesh và định dạng xuất file.</span>
                </div>
                """)
                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Cơ bản", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(label='Chế độ tạo hình',
                                            info='Turbo phù hợp đa số trường hợp; Fast dùng cho ảnh phức tạp; Standard ưu tiên chất lượng ổn định.',
                                            choices=['Turbo', 'Nhanh', 'Tiêu chuẩn'], value='Turbo')
                        decode_mode = gr.Radio(label='Độ chi tiết mesh',
                                               info='Điều chỉnh độ phân giải khi giải mã mesh.',
                                               choices=['Thấp', 'Tiêu chuẩn', 'Cao'],
                                               value='Tiêu chuẩn')
                    with gr.Tab('Nâng cao', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Xóa nền', min_width=100)
                            randomize_seed = gr.Checkbox(label="Seed ngẫu nhiên", value=True, min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Số bước suy luận')
                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Độ phân giải Octree')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Số chunk xử lý', min_width=100)
                    with gr.Tab("Xuất file", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='Định dạng', choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Giảm số mặt', value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Kèm texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Số mặt mục tiêu')
                        with gr.Row():
                            confirm_export = gr.Button(value="Chuẩn bị xuất", min_width=100, elem_classes='secondary-action')
                            file_export = gr.DownloadButton(label="Tải xuống", variant='primary',
                                                            interactive=False, min_width=100,
                                                            elem_classes='primary-action')

            with gr.Column(scale=7, elem_classes='studio-viewer'):
                gr.HTML("""
                <div class="panel-heading">
                    <strong>Xem trước mô hình</strong>
                    <span>Xoay, phóng to và kiểm tra mesh sau khi tạo.</span>
                </div>
                """)
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Mô hình đã tạo', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Kết quả')
                    with gr.Tab('Bản xuất file', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Kết quả xuất')
                    with gr.Tab('Thông số mesh', id='stats_panel'):
                        stats = gr.Json({}, label='Thông số')

            with gr.Column(scale=3 if MV_MODE else 2, elem_classes='studio-gallery'):
                gr.HTML("""
                <div class="panel-heading">
                    <strong>Thư viện mẫu</strong>
                    <span>Chọn nhanh ảnh mẫu để thử pipeline.</span>
                </div>
                """)
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Ảnh mẫu', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label='Mẫu ảnh', examples_per_page=18)

                    with gr.Tab('Mẫu văn bản', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label='Mẫu prompt', examples_per_page=18)
                    with gr.Tab('Mẫu nhiều góc', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label='Mẫu multiview', examples_per_page=6)

        gr.HTML(f"""
        <div class="studio-status">
        Model đang dùng: Shape ({args.model_path}/{args.subfolder}) | Texture ({'Hunyuan3D-2' if HAS_TEXTUREGEN else 'Không khả dụng'})
        </div>
        """)
        if not HAS_TEXTUREGEN:
            gr.HTML("""
            <div class="studio-warning">
                <b>Cảnh báo: </b>
                Tạo texture đang bị tắt do thiếu yêu cầu cài đặt.
                Xem <a href="https://github.com/Tencent/Hunyuan3D-2?tab=readme-ov-file#install-requirements">README.md</a> để bật lại.
            </div>
            """)
        if not args.enable_t23d:
            gr.HTML("""
            <div class="studio-warning">
                <b>Lưu ý: </b>
                Text to 3D đang tắt. Để bật, chạy `python gradio_app.py --enable_t23d`.
            </div>
            """)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        if HAS_T2I:
            tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value in ['Fast', 'Nhanh']:
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value in ['Low', 'Thấp']:
                return gr.update(value=196)
            elif value in ['Standard', 'Tiêu chuẩn']:
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                            textured=True)
            else:
                mesh = trimesh.load(file_out)
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                            textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--queue_max_size', type=int, default=16)
    parser.add_argument('--concurrency_limit', type=int, default=1)
    parser.add_argument('--timeout_keep_alive', type=int, default=120)
    args = parser.parse_args()

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 860
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div class='empty-viewer'>
      <div>
        <strong>Sẵn sàng tạo mô hình</strong>
        <span>Tải ảnh nguồn lên rồi tạo mesh để xem trước tại đây.</span>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            if args.low_vram_mode:
                texgen_worker.enable_model_cpu_offload()
            # Not help much, ignore for now.
            # if args.compile:
            #     texgen_worker.models['delight_model'].pipeline.unet.compile()
            #     texgen_worker.models['delight_model'].pipeline.vae.compile()
            #     texgen_worker.models['multiview_model'].pipeline.unet.compile()
            #     texgen_worker.models['multiview_model'].pipeline.vae.compile()
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e)
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    HAS_T2I = True
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=args.device)
        HAS_T2I = True

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=True,
        device=args.device,
    )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()

    @app.get("/health")
    async def health():
        gpu = {}
        if torch.cuda.is_available():
            gpu = {
                "device": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            }
        return {
            "status": "ok",
            "queue_max_size": args.queue_max_size,
            "concurrency_limit": args.concurrency_limit,
            "gpu": gpu,
        }

    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    demo.queue(
        status_update_rate="auto",
        max_size=args.queue_max_size,
        default_concurrency_limit=args.concurrency_limit,
    )
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        timeout_keep_alive=args.timeout_keep_alive,
        limit_concurrency=max(args.queue_max_size + 8, 32),
    )
