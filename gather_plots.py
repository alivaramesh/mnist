import sys, os, shutil


plots_dir = 'exps/plots'
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.mkdir(plots_dir)

for run in os.listdir('exps'):
    print(run)
    try:
        shutil.copy(os.path.join('exps',run,'results.jpg'),os.path.join(plots_dir,'{}.jpg'.format(run)))
    except IOError:
        pass