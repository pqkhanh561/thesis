from py4j.java_gateway import *
gateway = JavaGateway()

random = gateway.jvm.java.util.Random()
number1 = random.nextInt(10)
number2 = random.nextInt(10)

print(number1, number2)

addition_app = gateway.entry_point
value = addition_app.addition(number1, number2)
print(value)
