from transformers import BertTokenizer, BertForTokenClassification, DistilBertTokenizer, DistilBertForTokenClassification, RobertaTokenizer, RobertaForTokenClassification
import torch

class HybridConditionalModel:
    def __init__(self):
        """
        Initialize tokenizers and models for BERT, DistilBERT, and RoBERTa for ensemble predictions.
        """
        # Load BERT model and tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertForTokenClassification.from_pretrained("bert-base-uncased")

        # Load DistilBERT model and tokenizer
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

        # Load RoBERTa model and tokenizer
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaForTokenClassification.from_pretrained("roberta-base")

    def predict(self, text, max_length=512):
        """
        Make a prediction using BERT, DistilBERT, and RoBERTa, and return the combined predictions.
        
        :param text: Input text to predict labels for.
        :param max_length: Maximum token length for processing text.
        :return: Combined labels as a list.
        """
        # Process in chunks if text exceeds max length
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        combined_labels = []

        for chunk in chunks:
            # BERT
            bert_inputs = self.bert_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            bert_outputs = self.bert_model(**bert_inputs)
            bert_predictions = bert_outputs.logits.argmax(dim=-1)
            bert_labels = [self.bert_model.config.id2label[prediction.item()] for prediction in bert_predictions[0]]

            # DistilBERT
            distilbert_inputs = self.distilbert_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            distilbert_outputs = self.distilbert_model(**distilbert_inputs)
            distilbert_predictions = distilbert_outputs.logits.argmax(dim=-1)
            distilbert_labels = [self.distilbert_model.config.id2label[prediction.item()] for prediction in distilbert_predictions[0]]

            # RoBERTa
            roberta_inputs = self.roberta_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            roberta_outputs = self.roberta_model(**roberta_inputs)
            roberta_predictions = roberta_outputs.logits.argmax(dim=-1)
            roberta_labels = [self.roberta_model.config.id2label[prediction.item()] for prediction in roberta_predictions[0]]

            # Combine predictions using majority voting
            for bert_label, distilbert_label, roberta_label in zip(bert_labels, distilbert_labels, roberta_labels):
                label_counts = {bert_label: 0, distilbert_label: 0, roberta_label: 0}
                for label in [bert_label, distilbert_label, roberta_label]:
                    label_counts[label] += 1
                # Select the label with the most votes
                combined_labels.append(max(label_counts, key=label_counts.get))

        return combined_labels
