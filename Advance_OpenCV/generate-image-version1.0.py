import PIL
from PIL import Image, ImageDraw, ImageFont
def draw_text(text, size, fill=None):
    font = ImageFont.truetype('/home/cuong-nguyen/2016/Workspace/brexia/Novembre/CodeSources/captcha1/captcha/data/DejaVuSerif-Bold.ttf', size)
    size = font.getsize(text) # Returns the width and height of the given text, as a 2-tuple.
    #im = Image.new('RGBA', size, (0, 0, 0, 0)) # Create a blank image with the given size
    im = Image.open("background1.png")
    im=im.resize((400, 40),PIL.Image.ANTIALIAS)
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), text, font=font, fill=(0,0,0)) #Draw text
    return im
 
text=u"Hello"
img = draw_text(text, 30, (82, 124, 178))
img.save("out.png")

