FROM python:3.9

WORKDIR /opt
RUN pip install --upgrade pip
RUN pip install numpy==1.21.0 \
				pandas==1.3.0 \
				scikit-learn==0.24.2 \
				plotly==5.1.0 \
				matplotlib==3.4.2 \
				seaborn==0.11.1 \
				streamlit==0.84.1

WORKDIR /work

# 
# docker run -it -p 8888:8888 -v ~/git-portfolio/streamlit_PCA_WineClassification:/work <Image ID> bash
# streamlit run ***.py --server.port 8888
# --> localhost:8888

