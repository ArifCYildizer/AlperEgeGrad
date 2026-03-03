import cv2
import numpy as np
from rembg import remove

def change_back(background, img_rgba):
    """PNG (RGBA) olan görselin arka planını verilen background ile birleştirir."""
    background = cv2.resize(background, (img_rgba.shape[1], img_rgba.shape[0]), interpolation=cv2.INTER_AREA)
    res = np.copy(background)

    a = img_rgba[..., 3:].repeat(3, axis=2).astype("uint16")

    rgb = img_rgba[..., :3].astype("uint16")

    res = (res.astype("uint16") * (255 - a) // 255) + (rgb * a // 255)
    return res.astype("uint8")

def remove_bg(input_path, output_path="output.png"):
    with open(input_path, "rb") as i:
        input_bytes = i.read()
    output_bytes = remove(input_bytes)
    with open(output_path, "wb") as o:
        o.write(output_bytes)
    return output_path

if __name__ == "__main__":

    input_path = "images/input.jpg"      
    back_path  = "images/background.jpg" 

    output_path = "output.png"

    remove_bg(input_path, output_path)

    image_rgba = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if image_rgba is None:
        raise FileNotFoundError(f"Çıktı bulunamadı: {output_path}")
    if image_rgba.shape[2] != 4:
        raise ValueError("Çıktı görseli RGBA değil (alpha kanalı yok).")

    back = cv2.imread(back_path)
    if back is None:
        raise FileNotFoundError(f"Arka plan bulunamadı: {back_path}")


    result = change_back(back, image_rgba)

    cv2.imshow("Result", result)
    cv2.imwrite("result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()