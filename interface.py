import streamlit as st
from src.models_attention import Encoder,Decoder
from src.preprocess_obtain_dictclass import id1,id2
from src.evaluate_attn import testing,create_sentence
import torch

#Configuración de la página
st.set_page_config(page_title="dIAnaTranslator", layout="wide")


#Función para cargar los modelos
def load_models():
    #Instancio los modelos modelos
    enc1=Encoder(num_emb_id1=len(id1.word2index), emb_dim=512, hidden_size=256, num_layers_gru=2)
    dec1=Decoder(len(id2.index2word), 512, 256, len(id2.index2word), num_layers_gru=1, size_attn=256)
    
    enc2=Encoder(num_emb_id1=len(id1.word2index),emb_dim=512,hidden_size=256,num_layers_gru=2)
    dec2=Decoder(len(id2.index2word),512,256,len(id2.index2word),num_layers_gru=1,size_attn=256)

    #Cargo los pesos
    enc1.load_state_dict(torch.load("./models/encoder_attn6.pth", map_location='cpu'))
    dec1.load_state_dict(torch.load("./models/decoder_attn6.pth", map_location='cpu'))

    enc2.load_state_dict(torch.load("./models/encoder_attn7.pth", map_location='cpu'))
    dec2.load_state_dict(torch.load("./models/decoder_attn7.pth", map_location='cpu'))
    

    return enc1, dec1,enc2,dec2 #Retorno todos los modelos

enc1,dec1,enc2,dec2=load_models()

#Panel lateral
with st.sidebar:
    st.title("Configuración")
    st.markdown("---")
    st.write("**Dataset:** Kaggle English-Spanish")
    st.write("**Arquitectura:** Seq2Seq + Attention")
    st.info("Este traductor utiliza GRU layers con pesos entrenados desde cero.")


#Inicio interfaz
st.title("dIAnaTranslator")
st.markdown("### Traducción inteligente con mecanismo de atención")

#User input
text_user=st.text_area("Introduce la oración en inglés:", height=150)

if st.button("Traducir"):
    if text_user.strip()=="":
        st.warning("Por favor, introduce un texto para traducir.")
    else:
        col1,col2=st.columns(2)
        
        with col1:
            st.markdown("#### Modelo 1")
            with st.spinner('Procesando...'):
                tokens=testing(oracion=text_user, idioma_input=id1, modelo_encoder=enc1, modelo_decoder=dec1)
                sentence=create_sentence(tokens=tokens, idioma_target=id2)
                st.success(sentence)
                st.caption("Arquitectura: 2 GRU Layers + Attention")

        with col2:
            st.markdown("#### Modelo 2")
            with st.spinner('Procesando...'):
                tokens=testing(oracion=text_user, idioma_input=id1, modelo_encoder=enc2, modelo_decoder=dec2)
                sentence=create_sentence(tokens=tokens, idioma_target=id2)
                st.success(sentence)
                st.caption("Arquitectura: 2 GRU Layers + Attention")

st.markdown("---")
st.markdown("Hecho por dIAna")
