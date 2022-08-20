FROM python:3.10

ADD Ising2D.py .

RUN pip install --upgrade pip numpy quimb opt_einsum autoray jax jaxlib matplotlib networkx

CMD ["python", "./Ising2D.py"]
