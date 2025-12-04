import glob
from PIL import Image

def reformat(path, size):
    initial_image = Image.open(path, 'r')

    if len(initial_image.split()) > 3:
        image = Image.new("RGB", initial_image.size, (255, 255, 255))
        image.paste(initial_image, mask=initial_image.split()[3])
    else:
        image = initial_image

    width, height = image.size[0], image.size[1]

    if width != height:
        bigside = width if width > height else height
        background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
        background.paste(image, offset)

        return background.resize((size, size))
    else:

        return image.resize((size, size))    


if __name__ == '__main__':

    imgs = glob.glob('img/**.jpg')
    for img in imgs:
        new_img = reformat(img, size=512)
        new_img.save(img)





