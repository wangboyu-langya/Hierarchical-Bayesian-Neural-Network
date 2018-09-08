import smtplib
import sys
from email.MIMEMultipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.MIMEText import MIMEText
def mail(subject, pngs=False, dir_pic=None, txts=False, dir_txt=None, info='This is a test e-mail message.'):
    sender = 'xlhu13@fudan.edu.cn'
    password = 'this is not password'
    # receiver = 'hxianglong@gmail.com'
    receiver = ['xh1012@nyu.edu']
    server = 'mail.fudan.edu.cn'
    port = 465

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ','.join(receiver)
    msg['Subject'] = subject
    message = """\
     From: Experiment {0}
     To: Omnipotent Master {1}
     Subject: Lab Experiment Result

    {2}
    """.format(subject, 'hxl', info)
    msg.attach(MIMEText(message))

    if pngs:
        for png in pngs:
            fp = open(dir_pic + png, 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            img.add_header('Content-Disposition', 'attachment', filename=png)
            msg.attach(img)

    if txts:
        for txt in txts:
            fp = open(dir_txt + txt)
            t = MIMEText(fp.read())
            fp.close()
            # t = file(txt)
            # t = MIMEText(t.read())
            t.add_header('Content-Disposition', 'attachment', filename=txt)
            msg.attach(t)

    mailserver = smtplib.SMTP_SSL(server, port)
    mailserver.login(sender, password)
    try:
        mailserver.sendmail(sender, receiver, msg.as_string())
        print 'sent sucessfully'
    except Exception, exc:
        sys.exit("mail failed; %s" % str(exc))  # give a error message

    mailserver.quit()
