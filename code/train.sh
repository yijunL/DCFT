python train_demo.py \
    --train_iter 5000 \
    --trainN 5 --N 5 --K 1 --Q 1 \
    --val val_pubmed --test val_pubmed --adv pubmed_unsupervised \
    --lr 2e-5 \
    --model proto --encoder bert --hidden_size 768 --val_step 200 --random_seed 6 \
    --batch_size 1
