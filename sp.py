import os
import subprocess
directory = '/home/hxianglong/Projects/Hbnn-100'
screen = 'exp 2'
env = 'Hbnn'
experiment = 'test.py'
# subprocess.Popen('cd %s'%directory, shell=True)
# subprocess.Popen()
# bash = os.system()
# bash('bash -c \'cd %s\'' % directory)
# bash('bash -c \'screen -r %s\'' % screen)
# bash('bash -c \'source activate %s\'' % env)
os.system('bash -c \'cd %s\'' % directory)
os.system('bash -c \'screen -r %s\'' % screen)
os.system('bash -c \'source activate %s\'' % env)
# bash('bash -c \'python %s |& tee %s\'' % (experiment, experiment))
subprocess.Popen('python %s |& tee %s' % (experiment, experiment), shell=True)

# bash('bash -c \'screen -R %s\'' % directory)
