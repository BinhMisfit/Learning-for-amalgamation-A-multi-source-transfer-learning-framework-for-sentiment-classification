import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, BertPreTrainedModel

#class BertLSTMCNN(BertPreTrainedModel):
#    def __init__(self, config, hidden_size=128, dropout_prob=0.1):
#        super().__init__(config)

class BertLSTMCNN(nn.Module):
    def __init__(self, config_phobert, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1, maxlen=128, bert_only=False, phobert_only=False):
        super(BertLSTMCNN, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.num_labels = config_bert.num_labels
        self.bert_only=bert_only
        self.phobert_only=phobert_only

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        self.lstm = nn.LSTM(config_bert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(config_bert.hidden_size + 2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, config_bert.num_labels)

        self.phobert.init_weights()
        self.bert.init_weights()

    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # bert embeddings
        outputs_phobert = self.phobert(
            input_ids_phobert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        # MixtureOfExperts
        if self.bert_only is True:
            sequence_output = outputs_bert[0]
        elif self.phobert_only is True:
            sequence_output = outputs_phobert[0]
        else:
            method = 'SpMixtureOfExperts' #MixtureOfExperts, SpMixtureOfExperts, SpAttnMixtureOfExperts
            if method == 'MixtureOfExperts':
                inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

                inputs_gate = inputs_gate.detach()
                inputs_gate = self.gate_bn(inputs_gate)
                outputs_gate = self.gate(inputs_gate.float())
                outputs_gate_softmax = F.softmax(outputs_gate, dim=-1) ##torch.Size([bs, seqlen, 2])

                sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
                sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

            elif method == 'SpMixtureOfExperts':
                inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

                inputs_gate = inputs_gate.detach()
                outputs_gate = self.gate(inputs_gate.float())
                outputs_gate = self.gate_bn(outputs_gate)
                outputs_gate_softmax = F.sigmoid(outputs_gate)

                sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
                sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

            elif method == 'SpAttnMixtureOfExperts':
                pass



        lstm, _ = self.lstm(sequence_output)
        cat = torch.cat((lstm[:, :, :self.hidden_size], sequence_output, lstm[:, :, self.hidden_size:]), 2)
        linear = F.tanh(self.fc(cat)).permute(0, 2, 1)
        pool = F.max_pool1d(linear, linear.shape[2]).squeeze(2)

        pool = self.dropout(pool)
        logits = self.classifier(pool)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

#class BertGRUCNN(BertPreTrainedModel):
#    def __init__(self, config, hidden_size=128, dropout_prob=0.1):
#        super().__init__(config)

class BertGRUCNN(nn.Module):
    def __init__(self, config_phobert, config_bert, hidden_size=128, dropout_prob=0.1):
        super(BertGRUCNN, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = config_bert.num_labels

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        self.gru  = nn.GRU(config_bert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(config_bert.hidden_size + 2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, config_bert.num_labels)

        self.phobert.init_weights()
        self.bert.init_weights()

    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # bert embeddings
        outputs_phobert = self.phobert(
            input_ids_phobert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])


        # MixtureOfExperts
        method = 'SpMixtureOfExperts' #MixtureOfExperts, SpMixtureOfExperts, SpAttnMixtureOfExperts
        if method == 'MixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            inputs_gate = self.gate_bn(inputs_gate)
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate_softmax = F.softmax(outputs_gate, dim=-1) ##torch.Size([bs, seqlen, 2])

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpMixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate = self.gate_bn(outputs_gate)
            outputs_gate_softmax = F.sigmoid(outputs_gate)

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpAttnMixtureOfExperts':
            pass


        # final layers
        gru, _ = self.gru(sequence_output)
        cat = torch.cat((gru[:, :, :self.hidden_size], sequence_output, gru[:, :, self.hidden_size:]), 2)
        linear = F.tanh(self.fc(cat)).permute(0, 2, 1)
        pool = F.max_pool1d(linear, linear.shape[2]).squeeze(2)

        pool = self.dropout(pool)
        logits = self.classifier(pool)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
