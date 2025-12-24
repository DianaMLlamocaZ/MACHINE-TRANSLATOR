import torch

#Encoder
class Encoder(torch.nn.Module):
  def __init__(self,num_emb_id1,emb_dim,hidden_size,num_layers_gru):
    super().__init__()
    self.gru_layers=num_layers_gru
    self.emb=torch.nn.Embedding(num_embeddings=num_emb_id1,embedding_dim=emb_dim,padding_idx=0)
    self.gru=torch.nn.GRU(input_size=emb_dim,hidden_size=hidden_size,num_layers=num_layers_gru,batch_first=True)


  def forward(self,data):
    emb=self.emb(data)

    hidden_states,final_hidden_state=self.gru(emb)
    return hidden_states,final_hidden_state[int(self.gru_layers-1),:,:].unsqueeze(0)  #return encoder_output, 


#Attention Decoder
class BahdanauAttention(torch.nn.Module):
  def __init__(self,hidden_size_input,hidden_size_attn):
    super().__init__()
    self.queries=torch.nn.Linear(in_features=hidden_size_input,out_features=hidden_size_attn)
    self.key=torch.nn.Linear(in_features=hidden_size_input,out_features=hidden_size_attn)
    self.v=torch.nn.Linear(in_features=hidden_size_attn,out_features=1)
    
    self.tanh=torch.nn.Tanh()
    self.sm=torch.nn.Softmax(dim=1)

  def forward(self,h_s_encoder,h_s_decoder):
    qs=self.queries(h_s_encoder) #[batch,length,hidden_size_attn] --> [32,12,hidden_size_att] 
    k=self.key(h_s_decoder) #[batch,1,hidden_size_att] --> [32,1,hidden_size_att]
    s_no_norm=self.tanh(qs+k.permute(1,0,2))
    v_res=self.v(s_no_norm) #[32,12,1]
    
    weights=self.sm(v_res) #[32,12,1]
    context_vector=weights*qs
    context_vector_final=torch.sum(context_vector,dim=1)
   
    return context_vector_final


#Decoder
class Decoder(torch.nn.Module):
  def __init__(self,num_emb_id2,emb_dim,hidden_size,out_size,num_layers_gru,size_attn):
    super().__init__()
    self.emb=torch.nn.Embedding(num_embeddings=num_emb_id2,embedding_dim=emb_dim,padding_idx=0)
    self.gru=torch.nn.GRU(input_size=emb_dim+size_attn,hidden_size=hidden_size,num_layers=num_layers_gru,batch_first=True)
    self.ln1=torch.nn.Linear(in_features=hidden_size,out_features=hidden_size//2)
    self.relu=torch.nn.ReLU()
    self.ln2=torch.nn.Linear(in_features=hidden_size//2,out_features=out_size)
    self.dropout=torch.nn.Dropout(p=0.5)

    self.attention=BahdanauAttention(hidden_size_input=hidden_size,hidden_size_attn=size_attn)

  def forward(self,enc_outs,enc_hid_st,target):
    sequence_length=enc_outs.shape[1] #length de la secuencia --> 22 máx
    dec_hid_st=enc_hid_st #Hidden state inicial del decoder --> hidden state final de encoder
    

    preds=[] #Predicciones

    for i in range(sequence_length):
      #SOS Token
      if i==0:
        input_decoder=target[:,i].unsqueeze(1) #SOS token, 1 word

      #Los demás tokens
      else:
        #Attention
        attention_weights=self.attention(enc_outs,dec_hid_st).unsqueeze(1)
        
        input_embedding=self.dropout(self.emb(input_decoder))

        concat_input=torch.concat([attention_weights,input_embedding],dim=2)
        
        h_states,h_s_final=self.gru(concat_input,dec_hid_st)


        inputs_preds1=self.ln1(h_states.squeeze(1))
        inputs_preds2=self.ln2(self.relu(inputs_preds1))

       
        input_decoder=target[:,i].unsqueeze(1)

        dec_hid_st=h_s_final #Actualizo el hidden state del decoder


        preds.append(inputs_preds2.unsqueeze(1)) #AÑado 1 dim a partir del cual se concatenarán las predicciones completas por BATCH

    #Se retornan las PREDICCIONES POR BATCH
    return torch.cat(preds,dim=1)
