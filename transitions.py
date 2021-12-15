
class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence

        self.stack = ["ROOT"]
        self.buffer = [word for word in sentence]
        self.dependencies = []

    def parse_step(self, transition):
        if transition == 'S':
            self.stack.append(self.buffer.pop(0))
        elif transition == 'LA':
            dependency, head = self.stack[-2:]
            self.stack.pop(-2)
            self.dependencies.append((head, dependency))
        elif transition == 'RA':
            head, dependency = self.stack[-2:]
            self.stack.pop(-1)
            self.dependencies.append((head, dependency))

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)

        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]

    while unfinished_parses:
        minibatch_parses = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch_parses)

        for parse, transition in zip(minibatch_parses, transitions):
            parse.parse_step(transition)
            if len(parse.stack) < 2 and len(parse.buffer) < 1:
                unfinished_parses.remove(parse)

    dependencies = [p.dependencies for p in partial_parses]

    return dependencies
