#python visualize.py --eps 0.001 --to-show samples
#python visualize.py --eps 0.01 --to-show samples
#python visualize.py --eps 0.1 --to-show samples
#python visualize.py --eps 0.001 --attack-type linf --to-show histogram
#python visualize.py --eps 0.01 --attack-type linf --to-show histogram
#python visualize.py --eps 0.1 --attack-type linf --to-show histogram
#python visualize.py --eps 0.001 --attack-type l2 --to-show histogram
#python visualize.py --eps 0.01 --attack-type l2 --to-show histogram
#python visualize.py --eps 0.1 --attack-type l2 --to-show histogram
#python visualize.py --num-update-steps 25 --attack-type deepfool --to-show histogram
#python visualize.py --num-update-steps 50 --attack-type deepfool --to-show histogram
#python visualize.py --num-update-steps 75 --attack-type deepfool --to-show histogram
python visualize.py --eps 0.001 --to-show statistics --attack-type linf
python visualize.py --eps 0.01 --to-show statistics --attack-type linf
python visualize.py --eps 0.1 --to-show statistics --attack-type linf
python visualize.py --eps 0.001 --to-show statistics --attack-type l2
python visualize.py --eps 0.01 --to-show statistics --attack-type l2
python visualize.py --eps 0.1 --to-show statistics --attack-type l2
python visualize.py --num-update-steps 25 --to-show statistics --attack-type deepfool
python visualize.py --num-update-steps 50 --to-show statistics --attack-type deepfool
python visualize.py --num-update-steps 75 --to-show statistics --attack-type deepfool
