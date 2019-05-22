#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.001 \
#    --save-fname data/adversarial/linf_step10_eps0.001.pt \
#    --attack-type linf
#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.01 \
#    --save-fname data/adversarial/linf_step10_eps0.01.pt \
#    --attack-type linf
#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.1 \
#    --save-fname data/adversarial/linf_step10_eps0.1.pt \
#    --attack-type linf
#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.001 \
#    --save-fname data/adversarial/l2_step10_eps0.001.pt \
#    --attack-type l2
#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.01 \
#    --save-fname data/adversarial/l2_step10_eps0.01.pt \
#    --attack-type l2
#python adversarial.py --num-update-steps 10 --num-images 1000 \
#    --epsilon 0.1 \
#    --save-fname data/adversarial/l2_step10_eps0.1.pt \
#    --attack-type l2
python adversarial.py --num-update-steps 25 --num-images 1000 \
    --save-fname data/adversarial/deepfool_step25.pt \
    --attack-type deepfool \
    --bs 1
python adversarial.py --num-update-steps 50 --num-images 1000 \
    --save-fname data/adversarial/deepfool_step50.pt \
    --attack-type deepfool \
    --bs 1
python adversarial.py --num-update-steps 75 --num-images 1000 \
    --save-fname data/adversarial/deepfool_step75.pt \
    --attack-type deepfool \
    --bs 1
