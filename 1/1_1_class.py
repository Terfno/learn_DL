class Human:
  def __init__(self, name, gender):
    self.name = name
    self.gender = gender
    print("initialized.")
  
  def sayHello(self):
    print("Hello " + self.name + "!")

  def sayGoodbye(self):
    print("Good bye " + self.name + "!")

  def tellGender(self):
    print("I'm a " + self.gender + ".")

t = Human("terfno", "X-gender")
t.sayHello()
t.sayGoodbye()
