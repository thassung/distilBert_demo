from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_id,
    num_labels = 3,
    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2},
)

def create_student_model():
    configuration = teacher_model.config.to_dict()
    configuration['num_hidden_layers'] //= 2
    configuration = BertConfig.from_dict(configuration)
    model = type(teacher_model)(configuration)
    model.load_state_dict(torch.load('../model/Even layer model.pt'))
    model = model.to(device)
    return model

model = create_student_model()

def tokenize_function(sentence1, sentence2):
    args = (
        (sentence1,) if sentence2 is None else (sentence1, sentence2)
    )
    result = tokenizer(*args, max_length=128, truncation=True, return_tensors='pt')
    return result

def inference(sentence1, sentence2):
    tokenized_input = tokenize_function(sentence1, sentence2)
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    model.eval()
    with torch.no_grad():
        output_student = model(**tokenized_input)
    infer_student = output_student.logits.argmax(dim=-1).item()

    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    return id2label[infer_student]