"""CSST106 companion experiments. Run python labs.py --help.

Install: pip install opencv-contrib-python numpy matplotlib scikit-learn
         ultralytics torch torchvision pillow scikit-image
SURF needs an OpenCV source build with OPENCV_ENABLE_NONFREE=ON.
All fitted models and measured metrics in this script use actual computations.
Dataset/model downloads happen only when their relevant lab is invoked.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt


def gallery(images, titles, cols=3):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, im, title in zip(axes.flat, images, titles):
        ax.imshow(im, cmap='gray' if np.asarray(im).ndim == 2 else None)
        ax.set_title(title)
    fig.tight_layout()
    return fig


def load(path):
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f'Cannot load image: {path}')
    return im


def lab1(args):
    import cv2
    im = load(args.image)
    h, w, channels = im.shape
    x, y = min(w - 1, w // 2), min(h - 1, h // 2)
    print('Shape (height, width, channels):', im.shape)
    print(f'Pixel ({x},{y}) BGR:', im[y, x], 'RGB:', im[y, x, ::-1])
    edited = im.copy()
    edited[y, x] = (0, 0, 255)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    roi = im[h // 4: max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)]
    resized = cv2.resize(im, (max(1, w // 2), max(1, h // 2)))
    rotated = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    gallery([rgb(im), gray, rgb(edited), rgb(roi), rgb(resized), rgb(rotated)],
            ['Original', 'Grayscale', 'Edited center pixel (red)', 'ROI', 'Half size', '90 degrees'])


def lab2(args):
    import cv2
    im = cv2.cvtColor(load(args.image), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    canny = cv2.Canny(blur, args.low, args.high)
    gallery([im, magnitude, canny], ['Original', 'Sobel magnitude', 'Canny'])
    print('Sobel estimates gradient strength; Canny thins and connects thresholded edges.')


def lab3(args):
    import cv2
    im = load(args.image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(8, 8)).apply(gray)
    linear = np.clip(gray.astype(float) * args.alpha + args.beta, 0, 255).astype('uint8')
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    color = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    gallery([gray, eq, clahe, linear, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), color],
            ['Original gray', 'Global equalization', 'CLAHE', 'Linear alpha/beta', 'Original color', 'YUV luminance equalization'])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, a, title in zip(axes, [gray, eq, clahe, linear], ['Original', 'Global', 'CLAHE', 'Linear']):
        ax.hist(a.ravel(), bins=256, range=(0, 256), color='#54816c')
        ax.set_title(title)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Pixels')
    fig.tight_layout()


def lab4(args):
    import cv2
    a = cv2.cvtColor(load(args.image), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(load(args.second), cv2.COLOR_BGR2GRAY) if args.second else cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)
    detectors = {'SIFT': cv2.SIFT_create(), 'ORB': cv2.ORB_create(nfeatures=1000)}
    try:
        detectors['SURF'] = cv2.xfeatures2d.SURF_create(hessianThreshold=args.hessian, upright=args.upright)
    except (AttributeError, cv2.error) as exc:
        print('SURF unavailable. A source OpenCV build with contrib + OPENCV_ENABLE_NONFREE is required.')
        print(str(exc).splitlines()[0])
        print('Continuing with actual SIFT and ORB; no detector is mislabeled as SURF.')
    output, titles = [], []
    for name, detector in detectors.items():
        t0 = time.perf_counter()
        ka, da = detector.detectAndCompute(a, None)
        kb, db = detector.detectAndCompute(b, None)
        elapsed = time.perf_counter() - t0
        norm = cv2.NORM_HAMMING if name == 'ORB' else cv2.NORM_L2
        matches = []
        if da is not None and db is not None and len(db) >= 2:
            pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
        rich = cv2.drawKeypoints(a, ka, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        matched = cv2.drawMatches(a, ka, b, kb, sorted(matches, key=lambda m: m.distance)[:40], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        output.extend([cv2.cvtColor(rich, cv2.COLOR_BGR2RGB), cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)])
        titles.extend([f'{name}: {len(ka)} keypoints', f'{name}: {len(matches)} ratio-test matches'])
        print(name, 'keypoints:', len(ka), 'descriptor shape:', None if da is None else da.shape,
              'matching:', len(matches), f'detect+describe time: {elapsed:.4f}s')
    gallery(output, titles, cols=2)


def lab5(args):
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    data = fetch_lfw_people(min_faces_per_person=70, resize=.4)
    n, h, w = data.images.shape
    X, y = data.data, data.target
    print(f'LFW: {n} samples; {h} x {w} = {X.shape[1]} features; {len(data.target_names)} classes')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
    k = min(args.components, Xtrain.shape[0], Xtrain.shape[1])
    t0 = time.perf_counter()
    pca = PCA(n_components=k, svd_solver='randomized', whiten=True, random_state=42).fit(Xtrain)
    train_pca, test_pca = pca.transform(Xtrain), pca.transform(Xtest)
    print(f'PCA fit in {time.perf_counter() - t0:.3f}s; retained {pca.explained_variance_ratio_.sum():.2%}')
    eigenfaces = pca.components_.reshape(k, h, w)
    count = min(12, k)
    gallery(eigenfaces[:count], [f'Eigenface {i+1}' for i in range(count)], cols=4)
    if k >= 6:
        gallery(np.concatenate([eigenfaces[:3], eigenfaces[-3:]]),
                [f'First {i+1}' for i in range(3)] + [f'Last {k-2+i}' for i in range(3)])
    cumulative = pca.explained_variance_ratio_.cumsum()
    # Fit a separate full-spectrum PCA on TRAINING data to locate 90% even
    # when the requested 150 components do not retain 90%.
    full = PCA(n_components=None, svd_solver='full').fit(Xtrain)
    full_cum = full.explained_variance_ratio_.cumsum()
    ninety = int(np.searchsorted(full_cum, .9) + 1)
    print('Components required for 90% variance:', ninety)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(full_cum)+1), full_cum, label='Full training spectrum')
    plt.axhline(.9, color='gray', linestyle='--', label='90% variance')
    plt.axvline(ninety, color='green', linestyle='--', label=f'{ninety} components')
    plt.xlabel('Principal components'); plt.ylabel('Cumulative explained variance'); plt.legend(); plt.tight_layout()
    clf = SVC(kernel='rbf', class_weight='balanced').fit(train_pca, ytrain)
    pred = clf.predict(test_pca)
    report = classification_report(ytest, pred, labels=np.arange(len(data.target_names)),
                                   target_names=data.target_names, zero_division=0)
    print(report)
    Path('classification-report.txt').write_text(report, encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(ytest, pred, display_labels=data.target_names, xticks_rotation=90)
    plt.tight_layout()
    failed = np.flatnonzero(pred != ytest)[:12]
    if len(failed):
        gallery(Xtest[failed].reshape(-1, h, w),
                [f'True: {data.target_names[ytest[i]]}\nPred: {data.target_names[pred[i]]}' for i in failed], cols=4)
    else:
        print('No failures in this test split.')
    print('Whitening divides centered projections by sqrt(explained_variance_), giving unit training sample variance.')


def lab6(args):
    from ultralytics import YOLO
    if args.action == 'train':
        if not Path(args.data).exists():
            raise ValueError('Provide --data data.yaml with train/val paths and class names.')
        model = YOLO('yolo11n.pt')
        model.train(data=args.data, epochs=args.epochs, imgsz=640)
        print('Weights and measured training/validation metrics are in runs/detect/train (or the next numbered run).')
    elif args.action == 'val':
        metrics = YOLO(args.weights).val(data=args.data)
        print('mAP50:', metrics.box.map50, 'mAP50-95:', metrics.box.map,
              'precision:', metrics.box.mp, 'recall:', metrics.box.mr)
    elif args.action == 'dataset':
        root = Path('dataset')
        for directory in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            (root / directory).mkdir(parents=True, exist_ok=True)
        Path('data.yaml').write_text('path: ./dataset\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: car\n  2: bottle\n', encoding='utf-8')
        print('Created dataset folders and data.yaml. Add images and paired label .txt files.')
        print('Each label: class_id x_center y_center width height, normalized to [0,1].')
    else:
        model = YOLO(args.weights)
        source = 0 if args.webcam else args.image
        if source is None:
            raise ValueError('Use --image image.jpg or --webcam for YOLO inference.')
        results = model.predict(source=source, conf=args.confidence, iou=args.iou,
                                stream=True, show=args.webcam, save=not args.webcam)
        for result in results:
            for box in result.boxes:
                print(result.names[int(box.cls.item())], 'confidence:', float(box.conf.item()),
                      'xyxy:', box.xyxy.tolist(), 'YOLO xywh normalized:', box.xywhn.tolist())


def lab7(args):
    import cv2
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage
    im = load(args.image)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    # Flood fill in a copy; compare every neighbor to the seed (fixed range).
    filled = gray.copy()
    mask = np.zeros((gray.shape[0]+2, gray.shape[1]+2), dtype='uint8')
    seed = (min(gray.shape[1]-1, max(0, args.seed_x)), min(gray.shape[0]-1, max(0, args.seed_y)))
    cv2.floodFill(filled, mask, seed, 255, args.tolerance, args.tolerance,
                  cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
    region = mask[1:-1, 1:-1]
    distance = ndimage.distance_transform_edt(otsu > 0)
    coords = peak_local_max(distance, min_distance=15, labels=(otsu > 0).astype('uint8'))
    markers = np.zeros_like(gray, dtype='int32')
    for j, (y, x) in enumerate(coords):
        markers[y, x] = j + 1
    separated = watershed(-distance, markers, mask=otsu > 0)
    boundary = cv2.Canny(gray, args.low, args.high)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_regions = np.zeros_like(gray)
    cv2.drawContours(edge_regions, contours, -1, 255, thickness=cv2.FILLED)
    gallery([rgb, otsu, adaptive, region, separated, edge_regions],
            ['Original', 'Otsu', 'Adaptive', 'Region growing', 'Watershed labels', 'Filled edge contours'])
    if args.mask:
        truth = cv2.cvtColor(load(args.mask), cv2.COLOR_BGR2GRAY) > 127
        if truth.shape != gray.shape:
            raise ValueError('Ground truth must have the same dimensions as input.')
        for name, prediction in [('Otsu', otsu > 0), ('Adaptive', adaptive > 0),
                                  ('Region', region > 0), ('Edge contours', edge_regions > 0)]:
            union = np.logical_or(prediction, truth).sum()
            score = np.logical_and(prediction, truth).sum() / union if union else 1.0
            print(name, 'measured foreground IoU:', score)
    if args.semantic:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        from PIL import Image
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights).eval()
        batch = weights.transforms()(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            labels = model(batch)['out'][0].argmax(0).cpu().numpy()
        cmap = plt.get_cmap('tab20', len(weights.meta['categories']))
        labels = cv2.resize(labels.astype('uint8'), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(rgb); ax.imshow(labels, cmap=cmap, alpha=.55, vmin=0, vmax=len(weights.meta['categories'])-1)
        ax.axis('off'); ax.set_title('Actual pretrained DeepLabV3 semantic segmentation')
        present = np.unique(labels)
        import matplotlib.patches as patches
        ax.legend(handles=[patches.Patch(color=cmap(int(i)), label=weights.meta['categories'][i]) for i in present],
                  loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lab', type=int, required=True, choices=range(1, 8))
    parser.add_argument('--image'); parser.add_argument('--second'); parser.add_argument('--mask')
    parser.add_argument('--low', type=int, default=100); parser.add_argument('--high', type=int, default=200)
    parser.add_argument('--clip', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1.3); parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--hessian', type=float, default=400); parser.add_argument('--upright', action='store_true')
    parser.add_argument('--components', type=int, default=150)
    parser.add_argument('--action', choices=['infer', 'dataset', 'train', 'val'], default='infer')
    parser.add_argument('--data', default='data.yaml'); parser.add_argument('--weights', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--confidence', type=float, default=.5); parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--seed-x', type=int, default=0); parser.add_argument('--seed-y', type=int, default=0)
    parser.add_argument('--tolerance', type=int, default=25); parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()
    if args.lab not in [5, 6] and not args.image:
        parser.error('--image is required for this lab')
    if args.low < 0 or args.high <= args.low:
        parser.error('Use 0 <= low < high')
    if args.components < 1 or args.epochs < 1:
        parser.error('Components and epochs must be positive')
    [lab1, lab2, lab3, lab4, lab5, lab6, lab7][args.lab-1](args)
    if args.lab != 6:
        plt.show()


if __name__ == '__main__':
    main()
