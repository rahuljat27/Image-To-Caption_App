# Image Captioning Streamlit App

This is a polished Streamlit app to deploy your image captioning model.

## Structure
- `app.py` — Streamlit UI
- `model.py` — Model classes, preprocessing, loading, and captioning helper (extracted from your notebook)
- `checkpoint.pth` — Your trained weights (best epoch)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app tries to use your notebook's `VisionEncoder`, `Decoder`, `VisionEncoderDecoder`, and `generate_caption` directly.
- If your `generate_caption` signature differs, adjust `caption_image` in `model.py` accordingly.
- GPU is used automatically if available.
