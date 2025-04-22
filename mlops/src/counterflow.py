from metaflow import FlowSpec, step

class Counterflow(FlowSpec):
    @step
    def start(self):
        self.count = 0
        self.next(self.add)

    @step
    def add(self):
        print("The count is", self.count, "before incrementing")
        self.count += 1
        self.next(self.end)

    @step
    def end(self):
        self.count += 1
        print("Final count is", self.count)

if __name__ == '__main__':
    Counterflow()