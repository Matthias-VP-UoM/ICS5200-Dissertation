from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import requests
import streamlit as st
from PIL import Image
import io

from utils.explainability import explain_tabular_prediction, make_shap_bar_figure, explain_predictions
from utils.gradcam import create_gradcam_overlay
from utils.gradcam_explain import (
    find_hotspot_regions,
    regions_to_sentence,
    match_elements_to_hotspot,
    elements_to_sentence,
    draw_gradcam_debug_overlay,
    _bbox_to_page_pixels,
)
# from utils.gradcam_caption_gen import crop_hotspot, caption_with_ollama
from utils.hybrid_prediction import predict_hybrid
from utils.live_capture import capture_element_boxes, capture_page, create_driver, render_offline_html_for_elements
from utils.live_features import extract_live_features
from utils.live_prediction import LABEL_MAP, predict_accessibility


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "live_outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Application details
st.set_page_config(page_title="News Website Accessibility Evaluator", layout="wide")

st.title("Automated Accessibility Evaluation of News Websites")
st.write(
    "Enter a public news webpage URL to capture the current page, extract visual and DOM features, "
    "generate a prediction, and inspect SHAP and optional Grad-CAM explanations."
)

# Model configuration pane
with st.sidebar:
    st.header("Evaluation settings")

    prediction_mode = st.radio(
        "Prediction model",
        ["Tabular only", "Hybrid (visual embedding + tabular)"],
        help=(
            "Tabular only uses the DOM/screenshot-derived numeric features. "
            "Hybrid additionally feeds a CNN/ViT embedding of the screenshot "
            "into an enhanced Random Forest / Gradient Boosting model."
        ),
    )
    use_hybrid_model = prediction_mode.startswith("Hybrid")

    model_choice = st.selectbox("Model head", ["Random Forest", "Gradient Boosting"])
    show_shap = st.checkbox("Generate SHAP explanation", value=True)
    show_gradcam = st.checkbox("Generate Grad-CAM explanation", value=False)

    visual_architecture = st.selectbox(
        "Visual architecture",
        ["ResNet18", "ResNet50", "VGG16", "ViT_B_16", "Swin-T"],
        disabled=not (show_gradcam or use_hybrid_model),
        help="Used for Grad-CAM, and/or as the embedding backbone for the hybrid model.",
    )
    default_checkpoint = {
        "ResNet18": MODEL_DIR / "resnet18_model.pth",
        "ResNet50": MODEL_DIR / "resnet50_model.pth",
        "VGG16": MODEL_DIR / "vgg16_model.pth",
        "ViT_B_16": MODEL_DIR / "vit_b_16_model.pth",
        "Swin-T": MODEL_DIR / "swin_t_model.pth",
    }[visual_architecture]
    checkpoint_path = st.text_input(
        "Visual model checkpoint",
        value=str(default_checkpoint),
        disabled=not (show_gradcam or use_hybrid_model),
        help="The checkpoint must match the selected architecture and contain three output classes.",
    )
    # if use_hybrid_model:
        # st.caption(
        #     "Hybrid mode extracts this checkpoint's penultimate-layer embedding "
        #     "(head removed) and concatenates it with the tabular features before "
        #     "running the matching `enhanced_{architecture}_{rf|gb}` model. Confirm "
        #     "this matches how the hybrid models were trained — see the docstring "
        #     "in `utils/hybrid_prediction.py`."
        # )
    gradcam_method = st.selectbox(
        "Grad-CAM method",
        ["Grad-CAM", "Grad-CAM++", "Layer-CAM", "Score-CAM"],
        disabled=not (show_gradcam or use_hybrid_model),
        help="Used for Grad-CAM, and/or as the embedding backbone for the hybrid model.",
    )

gradcam_method_clean = str(gradcam_method).replace('-','').lower()

# Initialize session state variables
if "driver" not in st.session_state:
    st.session_state.driver = None

## Input Methods
input_method = st.radio(
    "Select your input method:",
    # ("Interactive Browser (Navigate manually)", "Automated URL Capture", "Upload Screenshot & HTML"),
    ("Interactive Browser (Navigate manually)", "Upload Screenshot & HTML"),
    horizontal=True
)
# Placeholders for evaluation requirements
screenshot_path = None
html_path = None
elements_path = None
final_url = "Manual Upload"

# Determine what input was pressed by the user
if input_method == "Interactive Browser (Navigate manually)":
    url = st.text_input("Webpage URL", placeholder="https://example.com/news/article")
    
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("1. Launch Browser Window", type="secondary"):
            if not url:
                st.warning("Please provide a URL.")
            else:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                # Close existing session if open
                if st.session_state.driver:
                    st.session_state.driver.quit()
                
                # Launch visible driver
                st.session_state.driver = create_driver(headless=False)
                st.session_state.driver.get(url)
                st.info("Browser launched! Scroll, dismiss popups, or navigate to the desired area in Chrome.")

    with col_nav2:
        run_evaluation = st.button("2. Capture Current View & Evaluate", type="primary")

    if run_evaluation:
        if not st.session_state.driver:
            st.error("Please launch the browser first!")
            execute_pipeline = False
        else:
            try:
                driver = st.session_state.driver
                screenshot_path = OUTPUT_DIR / "live_screenshot.png"
                html_path = OUTPUT_DIR / "live_page.html"

                # Capture what the user is currently looking at
                driver.save_screenshot(str(screenshot_path))
                html_path.write_text(driver.page_source, encoding="utf-8")

                # CROP SCREENSHOT
                with Image.open(screenshot_path) as img:
                    width, height = img.size
                    
                    # Define pixel margins to shave off (adjust these amounts to fit your needs)
                    X_CROP = 19
                    Y_CROP = 0

                    # Crop box format: (left, top, right, bottom)
                    crop_box = (
                        0, 
                        0, 
                        width - X_CROP, 
                        height - Y_CROP
                    )
                    
                    cropped_img = img.crop(crop_box)
                    cropped_img.save(screenshot_path)  # Overwrite saved image
                
                # Capture DOM element boxes while the live session is still open -
                # this is the most accurate source since it's the real rendered page,
                # not an offline approximation.
                elements = capture_element_boxes(driver)
                if elements:
                    elements_path = OUTPUT_DIR / "live_elements.json"
                    elements_path.write_text(json.dumps(elements), encoding="utf-8")

                final_url = driver.current_url
                execute_pipeline = True
                
                # Clean up driver
                driver.quit()
                st.session_state.driver = None
            except Exception as exc:
                st.error(f"Failed to capture active view: {exc}")
                execute_pipeline = False
# elif input_method == "Automated URL Capture":
#     url = st.text_input("Webpage URL", placeholder="https://example.com/news/article")
#     run_evaluation = st.button("Capture and evaluate webpage", type="primary")
#     capture_delay = 20

#     if run_evaluation:
#         if not url:
#             st.warning("Please provide a valid URL.")
#         else:
#             if not url.startswith(("http://", "https://")):
#                 url = "https://" + url
#             final_url = url
            
#             try:
#                 with st.status("Evaluating webpage", expanded=True) as status:
#                     st.write("Launching browser, loading requested webpage, and dismissing popups…")
                    
#                     # # 1. Fetch live screenshot through reliable API fallback
#                     # wait_path = f"wait/{capture_delay}/" if capture_delay > 0 else ""
#                     # api_url = f"https://image.thum.io/get/width/1280/crop/800/{wait_path}{url}"
#                     # response = requests.get(api_url, timeout=20+capture_delay)
#                     # if response.status_code != 200:
#                     #     raise Exception(f"Screenshot API returned status code {response.status_code}")

#                     # Run Selenium capture with popup dismissal
#                     screenshot_path, html_path, elements_path = capture_page(
#                         url=url, 
#                         output_dir=OUTPUT_DIR, 
#                         delay=capture_delay
#                     )
                    
#                     # # Save screenshot locally to match downstream expectations
#                     # screenshot_path = OUTPUT_DIR / "live_screenshot.png"
#                     # img = Image.open(io.BytesIO(response.content))
#                     # img.save(screenshot_path)
                    
#                     # # 2. Fetch HTML source code fallback
#                     # st.write("Fetching webpage DOM contents…")
#                     # html_response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
#                     # html_path = OUTPUT_DIR / "live_page.html"
#                     # html_path.write_text(html_response.text, encoding="utf-8")
                    
#                     # Re-route to the standard analysis pipeline
#                     st.write("Captured clean page layout & DOM structure successfully!")
#                     execute_pipeline = True
#             except Exception as exc:
#                 st.error(f"Capture failed: {exc}")
#                 execute_pipeline = False
else:
    # Upload Option
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_image = st.file_uploader("Upload Webpage Screenshot:", type=["png", "jpg", "jpeg"])
    with col_up2:
        uploaded_html = st.file_uploader("Upload Corresponding HTML DOM Source:", type=["html", "txt"])
    
    st.warning("⚠️ Disclaimer: Always ensure that the webpage screenshot is originated from the source HTML/DOM in order to maximise accuracy in predictions. Any differences in file sources could result in inaccuracies in the insights generated!")

    run_evaluation = st.button("Evaluate uploaded files", type="primary")
    
    if run_evaluation:
        if uploaded_image is None or uploaded_html is None:
            st.warning("Please upload both the screenshot image file AND the HTML file to proceed.")
            execute_pipeline = False
        else:
            try:
                with st.status("Processing uploaded files", expanded=True) as status:
                    # Save user uploaded image to outputs directory
                    screenshot_path = OUTPUT_DIR / "uploaded_screenshot.png"
                    img = Image.open(uploaded_image)
                    img.save(screenshot_path)
                    
                    # Save user uploaded HTML to outputs directory
                    html_path = OUTPUT_DIR / "uploaded_page.html"
                    html_path.write_bytes(uploaded_html.getvalue())

                    # Best-effort DOM element capture for Grad-CAM explanations (Step 2).
                    # This renders the uploaded HTML offline in a headless browser, so
                    # results may not perfectly match the uploaded screenshot if the
                    # page relied on relative-path CSS/JS/images. Failure here is
                    # non-fatal - the rest of the pipeline still runs without it.
                    st.write("Rendering uploaded HTML to map DOM elements for Grad-CAM explanations…")
                    elements = render_offline_html_for_elements(html_path, img.size)
                    if elements:
                        elements_path = OUTPUT_DIR / "uploaded_elements.json"
                        elements_path.write_text(json.dumps(elements), encoding="utf-8")
                        st.caption(f"Mapped {len(elements)} DOM elements (approximate - offline render).")
                    else:
                        elements_path = None
                        st.caption(
                            "Could not map DOM elements from the uploaded HTML "
                            "(this only affects the DOM-grounded Grad-CAM explanation - "
                            "prediction and SHAP results are unaffected)."
                        )

                    execute_pipeline = True
            except Exception as exc:
                st.error(f"File processing failed: {exc}")
                execute_pipeline = False

## Core Evaluation Pipeline Execution
# Runs seamlessly using the variables set up by either block above
if run_evaluation and 'execute_pipeline' in locals() and execute_pipeline:
    try:
        # Re-open status block to complete ML execution steps smoothly
        with st.status("Running feature analysis & model inference...", expanded=True) as status:
            st.write("Extracting screenshot and DOM features…")

            str_screenshot_path = str(screenshot_path)
            str_html_path = str(html_path)
            str_checkpoint_path = str(checkpoint_path)

            features = extract_live_features(str_screenshot_path, str_html_path)

            if use_hybrid_model:
                st.write(
                    f"Extracting {visual_architecture} embedding and generating "
                    f"hybrid {model_choice} prediction…"
                )
                result = predict_hybrid(
                    screenshot_path=str_screenshot_path,
                    tabular_features=features,
                    architecture=visual_architecture,
                    head_type=model_choice,
                    checkpoint_path=str_checkpoint_path,
                    label_map=LABEL_MAP,
                )
            else:
                st.write(f"Generating {model_choice} prediction…")
                result = predict_accessibility(features, model_choice)

            shap_df = None
            shap_figure = None
            if show_shap:
                st.write("Computing local SHAP feature contributions…")
                shap_df = explain_tabular_prediction(
                    result["model"], result["X_scaled"], result["prediction"]
                )
                if use_hybrid_model and shap_df is not None:
                    # Collapse the (largely uninterpretable) per-dimension visual
                    # embedding contributions into a single aggregate row so the
                    # chart stays readable, while keeping tabular features individual.
                    embedding_columns = set(result.get("embedding_columns", []))
                    is_embedding = shap_df["Feature"].isin(embedding_columns)
                    embedding_rows = shap_df[is_embedding]
                    tabular_rows = shap_df[~is_embedding]
                    if not embedding_rows.empty:
                        aggregate_row = pd.DataFrame(
                            {
                                "Feature": ["Visual embedding (aggregate)"],
                                "SHAP value": [embedding_rows["SHAP value"].sum()],
                                "Absolute impact": [embedding_rows["SHAP value"].sum().__abs__()],
                            }
                        )
                        shap_df = pd.concat([tabular_rows, aggregate_row], ignore_index=True).sort_values(
                            "Absolute impact", ascending=False
                        )
                shap_figure = make_shap_bar_figure(shap_df)

            gradcam_overlay = None
            visual_probabilities = None
            visual_class = None
            gradcam_error = None
            if show_gradcam:
                st.write("Computing Grad-CAM heatmap…")
                try:
                    gradcam_overlay, visual_probabilities, visual_class, gradcam_heatmap, cam_metrics = create_gradcam_overlay(
                        screenshot_path=str_screenshot_path,
                        checkpoint_path=str_checkpoint_path,
                        architecture=visual_architecture,
                        method=gradcam_method_clean
                    )
                except Exception as exc:
                    gradcam_error = str(exc)

            status.update(label="Evaluation complete", state="complete", expanded=False)
        
        # Stash everything the results UI needs into session_state instead of
        # rendering it inline here. Widgets rendered further down (like the
        # Grad-CAM debug checkbox) trigger a rerun of the whole script, and on
        # that rerun `run_evaluation` (a st.button() return value) goes back to
        # False, so this `if run_evaluation and ...:` block would be skipped
        # entirely and everything below it would vanish. Persisting the
        # results means the rendering block below can redraw them on every
        # rerun, regardless of whether the button was clicked this time.
        st.session_state.eval_results = {
            "final_url": final_url,
            "screenshot_path": screenshot_path,
            "elements_path": elements_path,
            "use_hybrid_model": use_hybrid_model,
            "model_choice": model_choice,
            "show_shap": show_shap,
            "show_gradcam": show_gradcam,
            "result": result,
            "shap_df": shap_df,
            "shap_figure": shap_figure,
            "gradcam_overlay": gradcam_overlay,
            "visual_probabilities": visual_probabilities,
            "visual_class": visual_class,
            "gradcam_heatmap": gradcam_heatmap if show_gradcam and not gradcam_error else None,
            "gradcam_error": gradcam_error,
        }

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

# Render results UI from session_state (not gated on run_evaluation) so that
# reruns triggered by widgets inside the tabs - e.g. the Grad-CAM debug
# overlay checkbox - don't wipe the results off the screen.
if "eval_results" in st.session_state:
    eval_results = st.session_state.eval_results
    try:
        final_url = eval_results["final_url"]
        screenshot_path = eval_results["screenshot_path"]
        elements_path = eval_results["elements_path"]
        use_hybrid_model = eval_results["use_hybrid_model"]
        model_choice = eval_results["model_choice"]
        show_shap = eval_results["show_shap"]
        show_gradcam = eval_results["show_gradcam"]
        result = eval_results["result"]
        shap_df = eval_results["shap_df"]
        shap_figure = eval_results["shap_figure"]
        gradcam_overlay = eval_results["gradcam_overlay"]
        visual_probabilities = eval_results["visual_probabilities"]
        visual_class = eval_results["visual_class"]
        gradcam_heatmap = eval_results["gradcam_heatmap"]
        gradcam_error = eval_results["gradcam_error"]

        # Output Results UI (Preserved exactly as your original code)
        st.caption(f"Source: {final_url}")

        metric_columns = st.columns(3)
        prediction_metric_label = "Hybrid prediction" if use_hybrid_model else "Tabular prediction"
        metric_columns[0].metric(prediction_metric_label, result["label"])
        metric_columns[1].metric("Confidence", f"{result['confidence']:.2%}")
        metric_columns[2].metric("Features extracted", len(result["feature_columns"]))

        screenshot_tab, features_tab, shap_tab, gradcam_tab = st.tabs(
            ["Captured webpage", "Features and probabilities", "SHAP", "Grad-CAM"]
        )

        with screenshot_tab:
            st.image(Image.open(screenshot_path), caption="Full-page screenshot", use_container_width=True)

        with features_tab:
            left, right = st.columns([1.25, 1])
            with left:
                st.subheader("Extracted Features")
                display_features = pd.DataFrame(
                    {"Feature": result["feature_columns"], "Value": result["X_raw"].iloc[0].values}
                )
                st.dataframe(display_features, use_container_width=True, hide_index=True)
            with right:
                st.subheader("Class Probabilities")
                probability_df = pd.DataFrame(
                    {
                        "Class": [LABEL_MAP.get(i, f"Class {i}") for i in range(len(result["probabilities"]))],
                        "Probability": result["probabilities"],
                    }
                )
                st.bar_chart(probability_df.set_index("Class"))

        with shap_tab:
            if not show_shap:
                st.info("Enable SHAP in the sidebar and run the evaluation again.")
            elif shap_figure is not None:
                st.pyplot(shap_figure)#, clear_figure=True)
                st.caption(
                    "Positive values push the model toward the predicted class; negative values push away from it."
                )
                if use_hybrid_model:
                    st.caption(
                        "Hybrid mode: the visual backbone's individual embedding dimensions are summed into "
                        "one 'Visual embedding (aggregate)' row so the chart stays readable; tabular "
                        "features are still shown individually."
                    )

                print(shap_df[["Feature", "SHAP value", "Absolute impact"]])

                # Display features which affect the prediction result
                st.subheader("Feature Prediction Effect Explanations")
                expls_pos_list, expls_neg_list, expls_zero_list = explain_predictions(shap_df[["Feature", "SHAP value", "Absolute impact"]], result["label"])

                if len(expls_pos_list) > 0:
                    st.html(f'<h3>Factors that pushed <i>toward</i> the result</h3>')

                    for expl in expls_pos_list:
                        st.write(expl)
                
                if len(expls_neg_list) > 0:
                    st.html(f'<h3>Factors that pushed <i>against</i> the result</h3>')

                    for expl in expls_neg_list:
                        st.write(expl)
                
                if len(expls_zero_list) > 0:
                    st.html(f'<h3>Factors that pushed **a negligible effect** on the result</h3>')
                
                if st.checkbox('View raw feature SHAP values?'):
                    st.dataframe(
                        shap_df[["Feature", "SHAP value", "Absolute impact"]],
                        use_container_width=True,
                        hide_index=True,
                    )

        with gradcam_tab:
            if not show_gradcam:
                st.info("Enable Grad-CAM in the sidebar and provide a compatible trained visual checkpoint.")
            elif gradcam_error:
                st.warning(
                    "The tabular evaluation completed, but Grad-CAM could not be generated. "
                    f"Details: {gradcam_error}"
                )
            else:
                col_original, col_cam = st.columns(2)
                with col_original:
                    st.image(Image.open(screenshot_path), caption="Original screenshot", use_container_width=True)
                with col_cam:
                    st.image(gradcam_overlay, caption="Grad-CAM overlay", use_container_width=True)
                
                visual_metric_columns = st.columns(2)
                visual_result = LABEL_MAP.get(visual_class, f"Class {visual_class}")
                visual_prob_max = visual_probabilities[visual_class]
                visual_metric_columns[0].metric("Visual model prediction", visual_result)
                visual_metric_columns[1].metric("Confidence", f"{visual_prob_max:.2%}")

                st.subheader("Grad-CAM Explanation")

                # Step 1: split the heatmap into separate connected regions rather than
                # one bbox spanning the min/max extent of ALL hot pixels combined -
                # that single-bbox approach balloons to cover most of the page whenever
                # attention is scattered across several disconnected areas.
                regions = find_hotspot_regions(gradcam_heatmap, max_regions=3)
                st.write(regions_to_sentence(regions, visual_result))

                page_image = Image.open(screenshot_path)

                # Step 2: DOM-grounded matches, computed separately per region
                matches_by_region = [[] for _ in regions]
                try:
                    with open(elements_path, "r", encoding="utf-8") as f:
                        page_elements = json.load(f)
                    for i, region in enumerate(regions):
                        region_matches = match_elements_to_hotspot(
                            region["bbox"],
                            gradcam_heatmap.shape,
                            page_elements,
                            screenshot_size=page_image.size,
                        )
                        matches_by_region[i] = region_matches
                        dom_sentence = elements_to_sentence(region_matches)
                        if dom_sentence:
                            st.write(f"**{region['region_label']}** region: {dom_sentence}")
                except Exception:
                    matches_by_region = [[] for _ in regions]

                # Debug/verification: draw each hotspot region (+ its matches) directly on
                # the actual screenshot, so mismatches are easy to spot visually rather
                # than reasoning about coordinates blind.
                if st.checkbox("Show Grad-CAM debug overlay (hotspot regions + matched elements)", value=False):
                    overlay_img = draw_gradcam_debug_overlay(page_image, regions, gradcam_heatmap.shape, matches_by_region)
                    st.image(overlay_img, caption="Each colour = a separate hotspot region, ranked by influence")

                # Step 3 (optional): local generative caption, off by default, clearly labelled
                # if regions and st.checkbox("Generate an AI visual description (experimental, requires Ollama)", value=False):
                #     top_region = regions[0]  # crop the single most influential region
                #     hotspot_px = _bbox_to_page_pixels(top_region["bbox"], gradcam_heatmap.shape, page_image.size)
                #     crop = crop_hotspot(page_image, hotspot_px)
                #     caption = caption_with_ollama(crop)
                #     if caption:
                #         st.caption(f"AI-generated description of the top region ({top_region['region_label']}), not independently verified:")
                #         st.write(caption)
                #     else:
                #         st.info("Could not reach a local Ollama vision model - is `ollama serve` running with a vision model pulled?")
                # visual_metric_columns[1].metric("Confidence", f"{result['confidence']:.2%}")
                # visual_metric_columns[2].metric("Features extracted", len(result["feature_columns"]))
                # st.metric("Visual model prediction", LABEL_MAP.get(visual_class, f"Class {visual_class}"))
                visual_probability_df = pd.DataFrame(
                    {
                        "Class": [LABEL_MAP.get(i, f"Class {i}") for i in range(len(visual_probabilities))],
                        "Probability": visual_probabilities,
                    }
                )
                st.bar_chart(visual_probability_df.set_index("Class"))
                if use_hybrid_model:
                    st.caption(
                        "Grad-CAM here explains the standalone fine-tuned classifier checkpoint. The headline "
                        "prediction above comes from the hybrid model, which reuses this checkpoint's backbone "
                        "as an embedding extractor but is otherwise a separate RF/GB classifier — so the two "
                        "predictions can legitimately disagree."
                    )
                else:
                    st.caption(
                        "Grad-CAM explains the selected visual model, whereas SHAP explains the selected tabular model. "
                        "They are separate explanations until a trained feature-fusion model is integrated."
                    )

    except Exception as exc:
        st.error(f"Displaying results failed: {exc}")