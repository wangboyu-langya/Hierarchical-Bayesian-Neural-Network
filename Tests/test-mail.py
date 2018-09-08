import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from Mail import mail

subject = 'Test Pics and Txts'
dir_pic = '/home/hxianglong/Projects/Hbnn-100/runs/test/imgs/'
pngs = ['overall.png', 'single.png']
dir_txt = '/home/hxianglong/Projects/Hbnn-100/'
txts = ['test.txt']
mail(subject, pngs, dir_pic, txts, dir_txt)
