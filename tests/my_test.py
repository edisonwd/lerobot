from transformers import AutoModel
model = AutoModel.from_pretrained('edison/act_policy')
print(model)