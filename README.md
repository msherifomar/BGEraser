# BGEraser

**Offline, privacy-preserving background removal for research presentations, clinical imagery, and educational materials.**

BGEraser is a desktop GUI that wraps the `rembg` library behind a controlled, reproducible workflow for academic and clinical users who cannot upload sensitive imagery to cloud services. It provides access to five segmentation backbones in a single interface, with alpha-matting controls for fine edges (hair, fabric, enamel margins, soft-tissue transitions), side-by-side preview, and direct export to PNG (transparent) or JPG (solid colour).

---

## Why BGEraser

Most background-removal tools send images to external servers. For researchers and clinicians handling:

- intraoral or prosthodontic photographs
- patient-identifiable clinical images
- proprietary laboratory specimens
- pre-publication figure material

cloud upload is often unacceptable. BGEraser runs entirely on the local machine after a one-time model download. All processing is offline.

---

## Features

- **Five selectable models**, side-by-side comparable on the same image:
  - BiRefNet (general purpose, best for hair and fine edges)
  - ISNet (fine-detail general use)
  - U2Net (fast, balanced)
  - U2Net Human (portraits)
  - SILUETA (tight, precise masks)
- **Alpha-matting controls**: foreground threshold, background threshold, erode size
- **Original / Split / Result** preview modes with live divider
- **Transparent PNG** or **solid-colour** output (PNG / JPG), with colour picker and quick presets
- **Dark, distraction-free UI** suitable for long figure-preparation sessions
- **No network connection required** after initial model download (~170 MB)

---

## Installation

```bash
pip install rembg Pillow numpy onnxruntime
```

For CUDA-enabled systems:

```bash
pip install rembg[gpu] Pillow numpy onnxruntime
```

---

## Usage

```bash
python BGEraser.py
```

1. **Open** an image (PNG, JPG, WEBP, BMP, TIFF)
2. **Select** a segmentation model (BiRefNet recommended as default)
3. **Toggle** alpha matting on for fine-edge work; adjust thresholds as needed
4. **Remove Background**
5. **Save** as transparent PNG or solid-colour PNG/JPG

First use of a given model will trigger a one-time download (~170 MB).

---

## When to use which model

| Model | Best for |
|---|---|
| **BiRefNet** | Portraits, hair, fine fibres, general-purpose accuracy |
| **ISNet** | Detailed subjects where edge fidelity matters more than speed |
| **U2Net** | Fast batch-style work with balanced quality |
| **U2Net Human** | Headshots and clinical portraits |
| **SILUETA** | Tight masks when halo artefacts must be minimised |

For clinical photography, start with BiRefNet with alpha matting enabled and adjust foreground threshold downward if edges appear over-eroded.

---

## System requirements

- Python 3.8+
- ~2 GB disk space (models + dependencies)
- Optional: CUDA-capable GPU for faster inference

---

## Citation

If you use BGEraser in a publication, presentation, or teaching material, please cite it. A `CITATION.cff` file is provided; a Zenodo DOI will be attached once the first release is archived.

---

## Author

**Dr. M S Omar**, BDS, MSc
Clinical Assistant Professor of Prosthodontics
Director, Digital Innovation Laboratory
Indiana University School of Dentistry
ORCID: [0009-0006-9198-3250](https://orcid.org/0009-0006-9198-3250)

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

BGEraser is built on the open-source [`rembg`](https://github.com/danielgatis/rembg) library by Daniel Gatis, which in turn wraps segmentation models from multiple academic research groups (BiRefNet, ISNet, U2Net, SILUETA). BGEraser contributes the workflow layer: model selection, alpha-matting exposure, preview comparison, and presentation-oriented export. The underlying segmentation is the work of those upstream projects.
