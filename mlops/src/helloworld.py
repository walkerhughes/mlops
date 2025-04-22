from metaflow import FlowSpec, step

class HelloWorldFlow(FlowSpec):
    @step
    def start(self):
        """Starting point"""
        print("This is start step")
        self.next(self.hello, self.hello2)

    @step
    def hello(self):
        """Say hi"""
        print("Hello World 1!")
        self.next(self.join)

    @step
    def hello2(self):
        """Say hi"""
        print("Hello World 2!")
        self.next(self.join)

    @step
    def join(self, inputs):
        print("Salutations have been said.")
        self.next(self.end)

    @step
    def end(self):
         """Finish line"""
         print("This is end step")

if __name__ == '__main__':
     HelloWorldFlow()