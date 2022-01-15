import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, BertPreTrainedModel


#class BertFC(BertPreTrainedModel):
#    def __init__(self, config, dropout_prob=0.1):
#        super().__init__(config)

class BertFC(nn.Module):
    def __init__(self, config_phobert, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1, maxlen=128, bert_only=False, phobert_only=False):
        super(BertFC, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.num_labels = self.config_bert.num_labels
        self.bert_only=bert_only
        self.phobert_only=phobert_only

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(config_bert.hidden_size*self.maxlen, config_bert.num_labels)
        #self.classifier = nn.Linear(config_bert.hidden_size, config_bert.num_labels)

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
        ) #torch.Size([bs, 768])

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
            sequence_output = outputs_bert[1]
        elif self.phobert_only is True:
            sequence_output = outputs_phobert[1]
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


        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

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
