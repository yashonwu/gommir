#!/bin/bash

# # ---------------------------------------------------------------------
# # gommir
# # ---------------------------------------------------------------------
# shoe
# # -----------------------
# # gommir-sl
 python models/gommir/train.py --data-type shoe --caption-model-dir usersim/checkpoints/shoe_show_attend_tell --model-dir results/shoe-sat-gommir --batch-size 128 --epochs 20 --lr 0.001 --top-k 3

# gommir-rl, gamma = 0.2
python models/gommir/train_rl.py --data-type shoe --caption-model-dir usersim/checkpoints/shoe_show_attend_tell --model-dir results/shoe-sat-gommir/gamma-0.2 --pretrained-model results/shoe-sat-gommir/sl-best.pt --batch-size 128 --epochs 20 --lr 1e-5 --top-k 3 --gamma 0.2

python models/gommir/eval.py --data-type shoe --caption-model-dir usersim/checkpoints/shoe_show_attend_tell --model-dir results/shoe-sat-gommir/gamma-0.2 --result-folder eval-rl-best --pretrained-model rl-best.pt --batch-size 128 --top-k 3

# # ---------------------------------------------------------------------
# # gommir
# # ---------------------------------------------------------------------
# dress
# # -----------------------
# # gommir-sl
python models/gommir/train.py --data-type dress --caption-model-dir usersim/checkpoints/dress_transformer --model-dir results/dress-tran-gommir --batch-size 128 --epochs 20 --lr 0.001 --top-k 3

# gommir-rl, gamma = 0.2
python models/gommir/train_rl.py --data-type dress --caption-model-dir usersim/checkpoints/dress_transformer --model-dir results/dress-tran-gommir/gamma-0.2 --pretrained-model results/dress-tran-gommir/sl-best.pt --batch-size 128 --epochs 20 --lr 1e-5 --top-k 3 --gamma 0.2

python models/gommir/eval.py --data-type dress --caption-model-dir usersim/checkpoints/dress_transformer --model-dir results/dress-tran-gommir/gamma-0.2 --result-folder eval-rl-best --pretrained-model rl-best.pt --batch-size 128 --top-k 3


# # ---------------------------------------------------------------------
# # gommir
# # ---------------------------------------------------------------------
# shirt
# # -----------------------
# # gommir-sl
python models/gommir/train.py --data-type shirt --caption-model-dir usersim/checkpoints/shirt_transformer --model-dir results/shirt-tran-gommir --batch-size 128 --epochs 20 --lr 0.001 --top-k 3

# gommir-rl, gamma = 0.2
python models/gommir/train_rl.py --data-type shirt --caption-model-dir usersim/checkpoints/shirt_transformer --model-dir results/shirt-tran-gommir/gamma-0.2 --pretrained-model results/shirt-tran-gommir/sl-best.pt --batch-size 128 --epochs 20 --lr 1e-5 --top-k 3 --gamma 0.2

python models/gommir/eval.py --data-type shirt --caption-model-dir usersim/checkpoints/shirt_transformer --model-dir results/shirt-tran-gommir/gamma-0.2 --result-folder eval-rl-best --pretrained-model rl-best.pt --batch-size 128 --top-k 3


# # ---------------------------------------------------------------------
# # gommir
# # ---------------------------------------------------------------------
# toptee
# # -----------------------
# # gommir-sl
python models/gommir/train.py --data-type toptee --caption-model-dir usersim/checkpoints/toptee_transformer --model-dir results/toptee-tran-gommir --batch-size 128 --epochs 20 --lr 0.001 --top-k 3

# gommir-rl, gamma = 0.2
python models/gommir/train_rl.py --data-type toptee --caption-model-dir usersim/checkpoints/toptee_transformer --model-dir results/toptee-tran-gommir/gamma-0.2 --pretrained-model results/toptee-tran-gommir/sl-best.pt --batch-size 128 --epochs 20 --lr 1e-5 --top-k 3 --gamma 0.2

python models/gommir/eval.py --data-type toptee --caption-model-dir usersim/checkpoints/toptee_transformer --model-dir results/toptee-tran-gommir/gamma-0.2 --result-folder eval-rl-best --pretrained-model rl-best.pt --batch-size 128 --top-k 3

