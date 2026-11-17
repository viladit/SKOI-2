import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MIN_MATCHES = 8


def make_reference(frame, sift, roi=None, quadrilateral=None):
    height, width = frame.shape[:2]
    if roi is not None and quadrilateral is not None:
        raise ValueError("Задайте либо прямоугольную область, либо четыре угла объекта")
    if quadrilateral is not None:
        corners = np.float32(quadrilateral).reshape(4, 2)
        if (not np.isfinite(corners).all() or not cv2.isContourConvex(corners)
                or cv2.contourArea(corners, oriented=True) < 100
                or np.any(corners < 0) or np.any(corners >= [width, height])):
            raise ValueError("Углы должны задавать выпуклый объект внутри кадра по часовой стрелке")
        roi = cv2.boundingRect(corners)
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
        raise ValueError("Область эталона должна находиться внутри первого кадра")
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    mask = None
    if quadrilateral is not None:
        corners = corners - np.float32([x, y])
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    else:
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < MIN_MATCHES:
        raise ValueError("В первом кадре слишком мало ключевых точек объекта")
    return keypoints, descriptors, corners, gray


def project_corners(homography, corners, frame_shape):
    if homography is None or not np.isfinite(homography).all():
        return None
    homogeneous = np.column_stack((corners, np.ones(4))) @ homography.T
    denominator = homogeneous[:, 2]
    if not (np.all(denominator > 1e-8) or np.all(denominator < -1e-8)):
        return None
    polygon = (homogeneous[:, :2] / denominator[:, None]).astype(np.float32)
    height, width = frame_shape[:2]
    if not np.isfinite(polygon).all() or np.abs(polygon).max() > 4 * max(width, height):
        return None
    if not cv2.isContourConvex(polygon):
        return None
    area = cv2.contourArea(polygon, oriented=True)
    if not 100 <= area <= 4 * width * height:
        return None
    viewport = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    visible_area, _ = cv2.intersectConvexConvex(polygon, viewport)
    if visible_area < 100:
        return None
    return polygon


def appearance_matches(frame, polygon, corners, reference_image):
    height, width = reference_image.shape
    inverse = cv2.getPerspectiveTransform(polygon, corners)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectified = cv2.warpPerspective(gray, inverse, (width, height))
    visible = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), inverse, (width, height))
    mask = np.zeros(reference_image.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    valid = (cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA) > 250)
    valid &= cv2.resize(visible, (128, 128), interpolation=cv2.INTER_AREA) > 250
    if valid.sum() < 128:
        return False
    source = cv2.resize(reference_image, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)[valid]
    target = cv2.resize(rectified, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)[valid]
    source -= source.mean()
    target -= target.mean()
    denominator = np.linalg.norm(source) * np.linalg.norm(target)
    return denominator > 1e-6 and np.dot(source, target) / denominator >= 0.3


def locate_object(frame, sift, matcher, reference):
    reference_keypoints, reference_descriptors, corners, reference_image = reference
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) < 2:
        return None
    pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
    candidates = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    matches = []
    used_reference = set()
    used_frame = set()
    for match in sorted(candidates, key=lambda item: item.distance):
        source = tuple(round(v, 1) for v in reference_keypoints[match.queryIdx].pt)
        target = tuple(round(v, 1) for v in keypoints[match.trainIdx].pt)
        if source not in used_reference and target not in used_frame:
            matches.append(match)
            used_reference.add(source)
            used_frame.add(target)
    if len(matches) < MIN_MATCHES:
        return None
    source_points = np.float32([reference_keypoints[m.queryIdx].pt for m in matches])
    target_points = np.float32([keypoints[m.trainIdx].pt for m in matches])
    homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
    if mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < MIN_MATCHES or inliers.mean() < 0.5:
        return None
    covered_area = cv2.contourArea(cv2.convexHull(source_points[inliers]))
    if covered_area < 0.03 * cv2.contourArea(corners):
        return None
    polygon = project_corners(homography, corners, frame.shape)
    if polygon is None or not appearance_matches(frame, polygon, corners, reference_image):
        return None
    return polygon


def update_reference(frame, sift, polygon, reference):
    corners = reference[2]
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
    if descriptors is None or len(keypoints) < MIN_MATCHES:
        return None
    inverse = cv2.getPerspectiveTransform(polygon, corners)
    points = np.float32([point.pt for point in keypoints]).reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
    for keypoint, point in zip(keypoints, points):
        keypoint.pt = tuple(map(float, point))
    return keypoints, descriptors, corners, reference[3]


def draw_result(frame, polygon, label):
    scale = max(0.6, min(frame.shape[:2]) / 720)
    thickness = max(1, round(scale))
    if polygon is None:
        text = "Tracking lost"
        anchor = (12, 12)
        color = (0, 180, 255)
    else:
        cv2.polylines(frame, [np.rint(polygon).astype(np.int32)], True, (0, 255, 0), max(2, round(2 * scale)), cv2.LINE_AA)
        text = label
        anchor = (polygon[:, 0].min(), polygon[:, 1].min())
        color = (0, 255, 0)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(np.clip(anchor[0], 4, max(4, frame.shape[1] - text_width - 5)))
    y = int(np.clip(anchor[1] - 8, text_height + 5, max(text_height + 5, frame.shape[0] - baseline - 5)))
    cv2.rectangle(frame, (x - 4, y - text_height - 4), (x + text_width + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def process_video(video_path, output_path, label="Object", roi=None, quadrilateral=None):
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"Видео не найдено {video_path}")
    if output_path == video_path or (output_path.exists() and output_path.samefile(video_path)):
        raise ValueError("Входное и выходное видео должны быть разными файлами")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Для результата укажите файл с расширением .mp4")
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Не удалось открыть видео {video_path}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Не удалось прочитать первый кадр")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Не удалось определить FPS видео")
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError("Для записи MP4 нужны чётные ширина и высота кадра")
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        cv2.setRNGSeed(0)
        reference = make_reference(frame, sift, roi, quadrilateral)
        recent_reference = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Не удалось создать видео {output_path}")
        total = 0
        detected = 0
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError("Размер кадров изменился внутри видео")
            polygon = locate_object(frame, sift, matcher, reference)
            if polygon is None and recent_reference is not None:
                polygon = locate_object(frame, sift, matcher, recent_reference)
            if polygon is not None:
                updated = update_reference(frame, sift, polygon, reference)
                if updated is not None:
                    recent_reference = updated
            detected += polygon is not None
            draw_result(frame, polygon, label)
            writer.write(frame)
            total += 1
            ok, frame = capture.read()
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected > 0 and total != expected:
            raise ValueError(f"Видео прочитано не полностью, получено {total} кадров из {expected}")
        return total, detected
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам")
    parser.add_argument("video", nargs="?", type=Path, default=BASE_DIR / "test-videos" / "mona-lisa.avi")
    parser.add_argument("-o", "--output", type=Path, help="Путь к результату .mp4")
    parser.add_argument("--label", default="Object", help="Подпись латиницей")
    region = parser.add_mutually_exclusive_group()
    region.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Объект в первом кадре, по умолчанию весь кадр")
    region.add_argument("--corners", nargs=8, type=int, metavar="N", help="Четыре угла объекта в первом кадре по часовой стрелке, начиная с левого верхнего")
    args = parser.parse_args()
    if not args.label.strip() or not args.label.isascii():
        parser.exit(1, "Используйте непустую подпись латиницей\n")
    output = args.output or BASE_DIR / "results" / f"{args.video.stem}_tracked.mp4"
    try:
        total, detected = process_video(args.video, output, args.label, args.roi, args.corners)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Ошибка обработки видео\n{error}\n")
    print(f"Сохранено {output.resolve()}")
    print(f"Кадров с рамкой {detected} из {total}. Без рамки {total - detected}.")


if __name__ == "__main__":
    main()
