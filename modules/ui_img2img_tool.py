"""Simplified image-to-image tool providing a streamlined interface for img2img generation."""
from contextlib import closing

import gradio as gr

from modules import shared, images
from modules.call_queue import wrap_gradio_gpu_call
from modules.processing import StableDiffusionProcessingImg2Img, process_images, fix_seed
from modules.shared import opts
from modules.ui_common import plaintext_to_html
from modules.ui_components import ResizeHandleRow


RESIZE_MODES = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]


def run_img2img_tool(
    image,
    prompt,
    negative_prompt,
    denoising_strength,
    cfg_scale,
    steps,
    width,
    height,
    resize_mode,
    seed,
):
    if image is None:
        return [], "", plaintext_to_html("Please upload an image first."), ""

    image = images.fix_image(image)

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        cfg_scale=cfg_scale,
        steps=steps,
        width=width,
        height=height,
        init_images=[image],
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
    )

    fix_seed(p)

    with closing(p):
        processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    output_images = [] if opts.do_not_show_images else processed.images

    return output_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def create_ui():
    with gr.Blocks(analytics_enabled=False) as img2img_tool_interface:
        with ResizeHandleRow(equal_height=False):
            with gr.Column(variant="compact", elem_id="img2img_tool_settings"):
                gr.HTML(
                    "<p style='margin-bottom:0.75em'>Transform any image using a text prompt. "
                    "Upload a reference image, describe what you want, then click <b>Generate</b>.</p>"
                )

                input_image = gr.Image(
                    label="Input Image",
                    elem_id="img2img_tool_input",
                    show_label=True,
                    sources=["upload", "clipboard"],
                    interactive=True,
                    type="pil",
                    image_mode="RGBA",
                    height=400,
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    elem_id="img2img_tool_prompt",
                    show_label=True,
                    lines=3,
                    placeholder="Describe what you want the output image to look like...",
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    elem_id="img2img_tool_negative_prompt",
                    show_label=True,
                    lines=2,
                    placeholder="What to exclude from the image (optional)...",
                )

                with gr.Accordion("Advanced Settings", open=False, elem_id="img2img_tool_advanced"):
                    denoising_strength = gr.Slider(
                        label="Image Strength",
                        elem_id="img2img_tool_denoising_strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.75,
                        info="0 = keep original, 1 = completely new image",
                    )

                    with gr.Row():
                        cfg_scale = gr.Slider(
                            label="Prompt Guidance (CFG Scale)",
                            elem_id="img2img_tool_cfg_scale",
                            minimum=1,
                            maximum=30,
                            step=0.5,
                            value=7.0,
                        )
                        steps = gr.Slider(
                            label="Sampling Steps",
                            elem_id="img2img_tool_steps",
                            minimum=1,
                            maximum=150,
                            step=1,
                            value=20,
                        )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            elem_id="img2img_tool_width",
                            minimum=64,
                            maximum=2048,
                            step=8,
                            value=512,
                        )
                        height = gr.Slider(
                            label="Height",
                            elem_id="img2img_tool_height",
                            minimum=64,
                            maximum=2048,
                            step=8,
                            value=512,
                        )

                    resize_mode = gr.Radio(
                        label="Resize Mode",
                        elem_id="img2img_tool_resize_mode",
                        choices=RESIZE_MODES,
                        value=RESIZE_MODES[0],
                        type="index",
                    )

                    seed = gr.Number(
                        label="Seed (-1 = random)",
                        elem_id="img2img_tool_seed",
                        value=-1,
                        precision=0,
                    )

                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="img2img_tool_generate",
                        variant="primary",
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        elem_id="img2img_tool_clear",
                        variant="secondary",
                    )

            with gr.Column(elem_id="img2img_tool_output_column"):
                output_gallery = gr.Gallery(
                    label="Result",
                    show_label=True,
                    elem_id="img2img_tool_gallery",
                    columns=1,
                    height=500,
                    object_fit="contain",
                    preview=True,
                )

                html_info = gr.HTML(elem_id="img2img_tool_html_info")
                generation_info = gr.Textbox(visible=False, elem_id="img2img_tool_generation_info")
                html_log = gr.HTML(elem_id="img2img_tool_html_log")

        generate_btn.click(
            fn=wrap_gradio_gpu_call(run_img2img_tool, extra_outputs=[None, "", ""]),
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                denoising_strength,
                cfg_scale,
                steps,
                width,
                height,
                resize_mode,
                seed,
            ],
            outputs=[output_gallery, generation_info, html_info, html_log],
            show_progress=False,
        )

        prompt.submit(
            fn=wrap_gradio_gpu_call(run_img2img_tool, extra_outputs=[None, "", ""]),
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                denoising_strength,
                cfg_scale,
                steps,
                width,
                height,
                resize_mode,
                seed,
            ],
            outputs=[output_gallery, generation_info, html_info, html_log],
            show_progress=False,
        )

        clear_btn.click(
            fn=lambda: (None, "", "", 0.75, 7.0, 20, 512, 512, RESIZE_MODES[0], -1),
            inputs=[],
            outputs=[
                input_image, prompt, negative_prompt,
                denoising_strength, cfg_scale, steps,
                width, height, resize_mode, seed,
            ],
        )

    return img2img_tool_interface
