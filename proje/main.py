import os
import cv2
import numpy as np
from rembg import remove


def change_back(background, img_rgba):
    """PNG (RGBA) olan görselin arka planını verilen background ile birleştirir."""
    background = cv2.resize(
        background,
        (img_rgba.shape[1], img_rgba.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    res = np.copy(background)

    alpha = img_rgba[..., 3:]
    alpha = np.repeat(alpha, 3, axis=2).astype("uint16")

    rgb = img_rgba[..., :3].astype("uint16")

    res = (res.astype("uint16") * (255 - alpha) // 255) + (rgb * alpha // 255)

    return res.astype("uint8")


def remove_bg(input_path, output_path="output.png"):
    """Verilen görselin arka planını kaldırır ve PNG olarak kaydeder."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Giriş görseli bulunamadı: {input_path}")

    with open(input_path, "rb") as i:
        input_bytes = i.read()

    output_bytes = remove(input_bytes)

    with open(output_path, "wb") as o:
        o.write(output_bytes)

    return output_path


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(BASE_DIR, "images", "input.jpg")
    back_path = os.path.join(BASE_DIR, "images", "background.jpg")
    output_path = os.path.join(BASE_DIR, "output.png")
    result_path = os.path.join(BASE_DIR, "result.jpg")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Giriş resmi bulunamadı:\n{input_path}\n\n"
            f"Lütfen 'images' klasörüne input.jpg dosyasını ekleyin."
        )

    if not os.path.exists(back_path):
        raise FileNotFoundError(
            f"Arka plan resmi bulunamadı:\n{back_path}\n\n"
            f"Lütfen 'images' klasörüne background.jpg dosyasını ekleyin."
        )

    remove_bg(input_path, output_path)

    image_rgba = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if image_rgba is None:
        raise FileNotFoundError(f"Çıktı görseli okunamadı: {output_path}")

    if len(image_rgba.shape) < 3 or image_rgba.shape[2] != 4:
        raise ValueError("Çıktı görseli RGBA değil, alpha kanalı bulunamadı.")

    back = cv2.imread(back_path)
    if back is None:
        raise FileNotFoundError(f"Arka plan görseli okunamadı: {back_path}")

    result = change_back(back, image_rgba)

    cv2.imwrite(result_path, result)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"İşlem tamamlandı. Sonuç dosyası: {result_path}")