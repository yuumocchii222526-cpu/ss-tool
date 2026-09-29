Palette Studio - Depth Anything V2 / Depth-aware DOF

- Existing Palette Studio features retained
- Automatic browser-side depth estimation with Depth Anything V2 Small (Transformers.js / ONNX)
- AI depth map is normalized to black=near / white=far
- Focus depth slider (0=near, 100=far)
- F-number simulation and depth-of-field width
- WebGL2 depth-aware lens blur with subject-mask protection
- MediaPipe foreground segmentation + correction brush
- Falls back to the previous lightweight progressive blur if AI depth or WebGL2 is unavailable
- Photos stay in the browser; only model files are downloaded on first use

GitHub Pages:
Upload index.html to the repository root. No Python backend is required.
The first AI-depth run downloads the quantized model files from Hugging Face/CDN, so it can take longer on iPhone.

Notes:
A screenshot does not contain FF14's real engine Z-buffer, so monocular depth is an estimate rather than exact physical scene depth.
