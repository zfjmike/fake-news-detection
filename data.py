import re
import torch

label_to_number = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

def dataset_to_variable(dataset, use_cuda):
	for i in range(len(dataset)):
		dataset[i].statement = torch.LongTensor(dataset[i].statement)
		dataset[i].subject = torch.LongTensor(dataset[i].subject)
		dataset[i].speaker = torch.LongTensor([dataset[i].speaker])
		dataset[i].speaker_pos = torch.LongTensor(dataset[i].speaker_pos)
		dataset[i].state = torch.LongTensor([dataset[i].state])
		dataset[i].party = torch.LongTensor([dataset[i].party])
		dataset[i].context = torch.LongTensor(dataset[i].context)
		if use_cuda:
			dataset[i].statement.cuda()
			dataset[i].subject.cuda()
			dataset[i].speaker.cuda()
			dataset[i].speaker_pos.cuda()
			dataset[i].state.cuda()
			dataset[i].party.cuda()
			dataset[i].context.cuda()

class DataSample:
	def __init__(self,
		label,
		statement,
		subject,
		speaker,
		speaker_pos,
		state,
		party,
		context):
		self.label = label_to_number.get(label, -1)
		self.statement = re.sub('[().]', '', statement).strip().split()
		while len(self.statement) < 5:
			self.statement.append('<no>')
		self.subject = subject.strip().split(',')
		self.speaker = speaker
		self.speaker_pos = speaker_pos.strip().split()
		self.state = state
		self.party = party
		self.context = context.strip().split()

		if len(self.statement) == 0:
			self.statement = ['<no>']
		if len(self.subject) == 0:
			self.subject = ['<no>']
		if len(self.speaker) == 0:
			self.speaker = '<no>'
		if len(self.speaker_pos) == 0:
			self.speaker_pos = ['<no>']
		if len(self.state) == 0:
			self.state = '<no>'
		if len(self.party) == 0:
			self.party = '<no>'
		if len(self.context) == 0:
			self.context = ['<no>']

def count_in_vocab(dict, word):
	if word not in dict:
		dict[word] = len(dict)
		return dict[word]
	else:
		return dict[word]

def train_data_prepare(train_filename):
	print("Preparing data from: " + train_filename)
	train_file = open(train_filename, 'rb')

	lines = train_file.read()
	lines = lines.decode("utf-8")
	train_samples = []
	statement_word2num = {'<unk>' : 0}
	subject_word2num = {'<unk>' : 0}
	speaker_word2num = {'<unk>' : 0}
	speaker_pos_word2num = {'<unk>' : 0}
	state_word2num = {'<unk>' : 0}
	party_word2num = {'<unk>' : 0}
	context_word2num = {'<unk>' : 0}

	for line in lines.strip().split('\n'):
		tmp = line.strip().split('\t')
		while len(tmp) != 8:
			tmp.append('')
		p = DataSample(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7])

		for i in range(len(p.statement)):
			p.statement[i] = count_in_vocab(statement_word2num, p.statement[i])
		for i in range(len(p.subject)):
			p.subject[i] = count_in_vocab(subject_word2num, p.subject[i])
		p.speaker = count_in_vocab(speaker_word2num, p.speaker)
		for i in range(len(p.speaker_pos)):
			p.speaker_pos[i] = count_in_vocab(speaker_pos_word2num, p.speaker_pos[i])
		p.state = count_in_vocab(state_word2num, p.state)
		p.party = count_in_vocab(party_word2num, p.party)
		for i in range(len(p.context)):
			p.context[i] = count_in_vocab(context_word2num, p.context[i])
		
		train_samples.append(p)

	word2num = [statement_word2num,
				subject_word2num,
				speaker_word2num,
				speaker_pos_word2num,
				state_word2num,
				party_word2num,
				context_word2num]

	print("  "+str(len(train_samples))+" samples")

	print("  Statement Vocabulary Size: " + str(len(statement_word2num)))
	print("  Subject Vocabulary Size: " + str(len(subject_word2num)))
	print("  Speaker Vocabulary Size: " + str(len(speaker_word2num)))
	print("  Speaker Position Vocabulary Size: " + str(len(speaker_pos_word2num)))
	print("  State Vocabulary Size: " + str(len(state_word2num)))
	print("  Party Vocabulary Size: " + str(len(party_word2num)))
	print("  Context Vocabulary Size: " + str(len(context_word2num)))

	return train_samples, word2num


