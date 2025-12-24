from preprocess_obtain_dictclass import normalizeString,tokenize_sentence
import torch


def testing(oracion,idioma_input,modelo_encoder,modelo_decoder):
  oracion_nm=normalizeString(s=oracion)
  oracion_norm_tokens=tokenize_sentence(idioma=idioma_input,oracion_norm=oracion_nm)
  tensor_oracion=torch.tensor(oracion_norm_tokens).unsqueeze(0)

  traduccion_tokens=[]

  #Encoder final hidden state
  modelo_encoder.eval()
  with torch.no_grad():
    hid_states,final_hid_state=modelo_encoder(tensor_oracion)
    
  #Decoder
  modelo_decoder.eval()
  dec_hid_state=final_hid_state
  with torch.no_grad():
    input_decoder=torch.tensor([1])

    while input_decoder.item()!=2:

      attn_weights=modelo_decoder.attention(hid_states,dec_hid_state).unsqueeze(1)
      
      input_emb_dec=modelo_decoder.emb(input_decoder.unsqueeze(0))
     
      input_decoder_concat=torch.concat([attn_weights,input_emb_dec],dim=2)
      
      h_s_outs,final_h_s=modelo_decoder.gru(input_decoder_concat,dec_hid_state)
      
      logits_pred1=modelo_decoder.ln1(h_s_outs.squeeze(1))
      logits_pred2=modelo_decoder.ln2(modelo_decoder.relu(logits_pred1))
        
      input_decoder=torch.argmax(logits_pred2,dim=1)
      
      dec_hid_state=final_h_s
      
      traduccion_tokens.append(input_decoder.item())
      

      if input_decoder.item()==2:
          return traduccion_tokens
      
def create_sentence(tokens,idioma_target):
  palabras=""
  for token in tokens:
    word=idioma_target.index2word[token]
    if word!="eos":
        palabras+=word +" "

  return palabras
