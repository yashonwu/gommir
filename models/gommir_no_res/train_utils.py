import torch
import math
import time

def train_sl(epoch, optimizer, triplet_loss, model, user, ranker, batch_size, top_k, turns):
    print('train epoch #{}'.format(epoch))
    model.train()
    triplet_loss.train()
    split= 'train'
    # train / test
    all_input = user.train_feature
    dialog_turns = turns

    user_img_idx = torch.LongTensor(batch_size)
    top_k_act_img_idx = torch.LongTensor(batch_size,top_k)
    neg_img_idx = torch.LongTensor(batch_size)
    total_batch = math.ceil(all_input.size(0) / batch_size)

    for batch_idx in range(1, total_batch + 1):
        start = time.time()

        # sample target images
        user.sample_idx(user_img_idx, split)
        # user.sample_target_idx(user_img_idx, split, batch_idx, batch_size, total_batch)
        
        # sample initial top-k recommendation
        user.sample_k_idx(top_k_act_img_idx, split, top_k)

        # update item embeddings
        feat = model.update_rep(all_input)
        ranker.update_rep(feat)
        
        # clear history
        model.init_hist()
        
        outs = []

        for k in range(dialog_turns):
            # non-verbal relevance feedback: like- the most similar item to the target, dislikes- the rest items
            p_act_img_idx, p_position, n_act_img_idx, n_position = ranker.nearest_neighbor_selector(user_img_idx, top_k_act_img_idx)
            act_input = all_input[p_act_img_idx]
            
            # verbal relevance feedback
            txt_input,_ = user.get_feedback(p_act_img_idx, p_position, user_img_idx, split)
            
            # state tracking
            if torch.cuda.is_available():
                act_input = act_input.cuda()
                txt_input = txt_input.cuda()
            state, combined_emb = model.merge_forward(act_input, txt_input)
            
            # sampling for the next turn
            new_top_k_act_img_idx = ranker.k_nearest_neighbors(state.data,K=top_k)

            # triplet loss
            # ranking_candidate = ranker.compute_rank(state.data, user_img_idx)
            user_emb = ranker.feat[user_img_idx]
            user.sample_idx(neg_img_idx, split)
            neg_emb = ranker.feat[neg_img_idx]
            # loss = triplet_loss.forward(state, user_emb, neg_emb)
            loss = triplet_loss.forward(state, user_emb, neg_emb) + triplet_loss.forward(combined_emb, user_emb, neg_emb)
            
            outs.append(loss)

            ## option 1: random new actions
            user.sample_k_idx(top_k_act_img_idx, split, top_k)
            
            # option 2: next action
            # top_k_act_img_idx = new_top_k_act_img_idx

        # finish dialog and update model parameters
        optimizer.zero_grad()
        outs = torch.stack(outs, dim=0).mean()
        outs.backward()
        optimizer.step()

        end = time.time()
        print('batch_idx:', batch_idx, '/', total_batch, ', time elapsed:{:.2f}'.format(end - start))
    return