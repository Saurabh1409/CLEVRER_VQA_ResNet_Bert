import numpy as np
import pandas as pd
import json
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #model params
    parser.add_argument('--Des_json', type=str)
    parser.add_argument('--Ndes_json', type=str)
    parser.add_argument('--test_gt',type=str)
    #parser.add_argument('--model',type=str)
    args = parser.parse_args()

    df_ndes = pd.read_json(args.Ndes_json)
    df_des = pd.read_json(args.Des_json)
    df_des["Pred_answer"] = 0

    for j,i in enumerate(df_des["Pred"]):
        df_des["Pred_answer"][j] = np.argmax(i)

    df_des_c = pd.DataFrame(df_des)
    for i in range(len(df_des)):
        df_des_c.loc[i, "Scene"]=df_des.iloc[i]["Scene"][0]
        df_des_c.loc[i, "Q_id"]=df_des.iloc[i]["Q_id"][0]

    for k,i in enumerate(df_ndes["Pred"]):
        for j,v in enumerate(i):
            if v>=0.5:
                df_ndes["Pred"][k][j]=1
                
            else:
                df_ndes["Pred"][k][j]=0

    df_ndes_c = pd.DataFrame(df_ndes)
    for i in range(len(df_ndes)):
        df_ndes_c.loc[i, "Scene"]=df_ndes.iloc[i]["Scene"][0]
        df_ndes_c.loc[i, "Q_id"]=df_ndes.iloc[i]["Q_id"][0]
        #df_ndes_c.loc[i, "Pred"]=df_ndes.iloc[i]["Pred"][0]

    idx2word = {0: "sphere", 1: "blue", 2: "rubber", 3: "3", 4: "2", 5: "no", 6: "yellow", 7: "cube", 8: "1", 9: "gray", 10: "metal", 11: "yes", 12: "cylinder", 13: "0", 14: "4", 15: "brown", 16: "purple", 17: "red", 18: "green", 19: "cyan", 20: "5"}

    final = []
    test = pd.read_json(args.test_gt)
    temp_test = test.iloc[15000-15000]["questions"]
    for scene in range(15000,20000):
        #print(df_des["Scene"][scene-15000][0])
        temp_des=df_des_c[df_des_c['Scene']==scene]
        temp_ndes=df_ndes_c[df_ndes_c['Scene']==scene]
        temp_test = test.iloc[scene-15000]["questions"]
        ques=[]
        for idx in range(len(temp_des)):
            ques.append({
                "question_id" : temp_des.iloc[idx]["Q_id"],
                "answer" : idx2word[temp_des.iloc[idx]["Pred_answer"]]
            })
        for idx in range(len(temp_ndes)):
            ch_list=[]
            for i,ch in enumerate(temp_ndes.iloc[idx,2]):
                if i < len(temp_test[temp_ndes.iloc[idx]["Q_id"]]["choices"]):
                    if ch == 0:
                        ch_list.append({"choice_id":i,"answer":"wrong"})
                    else:
                        ch_list.append({"choice_id":i,"answer":"correct"})
            ques.append({
                "question_id" : temp_ndes.iloc[idx]["Q_id"],
                "choices" : ch_list
            })    
            
        cur={
            "scene_index" : scene,
            "questions" : ques
        }
        final.append(cur)

    with open('CLEV_PART.json', "w") as f:
        json.dump(final, f)

# python json_gen.py --Des_json Des.json --Ndes_json Ndes.json --test_gt Dataloader/test.json