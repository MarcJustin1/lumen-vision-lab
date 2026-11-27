# Lumen Computer Vision Laboratory

A React 19 and Vite academic simulation based on the seven supplied CSST106 laboratory documents. Source documents remain unchanged in `../docs`.

## Run locally

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. To build and preview the production app:

```sh
npm test
npm run build
npm run preview
```

## Coverage and computation boundaries

| Laboratory | Browser experience | Python companion |
| --- | --- | --- |
| Images | Pixel inspection/editing, RGB/BGR, grayscale, ROI, nearest-neighbor resizing, 90° rotations | OpenCV image operations |
| Edges | Gaussian 3×3, Sobel magnitude, Canny suppression and hysteresis | OpenCV Sobel and Canny |
| Histograms | Histograms, global equalization, approximate luminance-only color equalization, interpolated tile CLAHE, alpha/beta contrast | OpenCV YUV luminance equalization and CLAHE |
| SURF | Clearly labeled finite-difference detector approximations, simplified 64-value descriptors, rotated-image matches, upright mode | Actual SURF where available, actual SIFT and ORB, descriptor matching |
| PCA | Actual Gram-matrix PCA, training-only centering, whitening, reconstruction, 12 eigenfaces, variance curve, balanced one-vs-rest RBF SVM and measured reports on 48 synthetic faces | LFW ≥70 faces/person, 80/20 split, 150 components, randomized PCA, whitening, balanced RBF SVC, full-spectrum 90% analysis, failures |
| YOLO11 | Fixed illustrative detection candidates with real confidence/NMS filtering, manual annotations and normalized YOLO exports, dataset YAML, illustrative training curves | Actual pretrained inference, webcam, dataset scaffold, custom training and validation |
| Segmentation | Otsu, local adaptive thresholding, connected seed growing, simplified distance-marker watershed, Canny boundaries, fixed illustrative semantic overlay, optional measured binary IoU | OpenCV/scikit-image segmentation, filled contours, actual pretrained DeepLabV3, binary IoU |

The browser does **not** execute pretrained neural models, train YOLO, use LFW, or implement production SURF/SIFT/ORB. It labels all such illustrations explicitly. Results are educational; no synthetic accuracy or training curve is presented as a real-dataset benchmark. The browser watershed uses multi-source foreground flooding rather than a full topographic watershed.

## Notebook and exports

Observations, answers, and the latest 100 runs are saved in browser localStorage. Images are not persisted across reloads. Uploaded images are resized to fit 640×480 before processing and stay in the browser. Reports export as standalone HTML with current experiment images and can be printed to PDF. PNG exports contain the processed image; feature/box coordinates are also available through the current result export. JSON preserves run parameters, provenance, notes, and metrics. Keep exported YOLO labels paired with their corresponding processed-size input image.

## Python experiments

The website provides `public/labs.py` for download. Run it outside the browser:

```sh
pip install opencv-contrib-python numpy matplotlib scikit-learn ultralytics torch torchvision pillow scikit-image
python public/labs.py --lab 1 --image input.jpg
python public/labs.py --lab 4 --image input.jpg --second second.jpg --hessian 400
python public/labs.py --lab 5
python public/labs.py --lab 6 --action dataset
python public/labs.py --lab 6 --image input.jpg
python public/labs.py --lab 6 --webcam
python public/labs.py --lab 6 --action train --data data.yaml --epochs 100
python public/labs.py --lab 6 --action val --weights runs/detect/train/weights/best.pt --data data.yaml
python public/labs.py --lab 7 --image input.jpg --semantic --mask truth.png
```

SURF requires a suitable OpenCV build with contrib and `OPENCV_ENABLE_NONFREE=ON`; ordinary wheels may not support it. The script reports unavailability and still compares real SIFT/ORB without substituting detector identities. LFW and pretrained weights require downloads and enough local compute. The Python companion is syntax-checked; model downloads/training are not executed as part of building the React app.

The YOLO file is named YOLOV8 in the supplied docs but its body specifies YOLOv11; the implementation follows the body and uses the official `yolo11n.pt` checkpoint name.

## Interface

Responsive white and sage laboratory dashboard with a persistent experiment sidebar. The supplied Pinterest pin was inaccessible to both web fetching and the available browser connection, so visual fidelity to it could not be verified.

Official references: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [YOLO11](https://docs.ultralytics.com/models/yolo11/), [OpenCV](https://docs.opencv.org/4.x/).
