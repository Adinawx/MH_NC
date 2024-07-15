# Basic Concepts

SimPy is a discrete-event simulation library.

*Processes* []{dir="rtl"}-- Models the behavior of active components
(like vehicles, customers, or messages). They are described by simple
Python [generators](http://docs.python.org/3/reference/expressions.html#yieldexpr).
You can call them *process function* or *process method*, depending on
whether it is a normal *function* or method of a class. During their
lifetime, they create events and yield them to wait for them to occur.
When a process yields an *event*, the process gets *suspended*.
SimPy *resumes* the process when the event occurs (we say that the event
is *processed*). Multiple processes can wait for the same event. SimPy
resumes them in the same order in which they yielded that event.

*Environment* -- where processes live.

*Event* -- The way processes interact with the environment and with
themselves. An important event type is
the [**Timeout**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Timeout).
Events of this type occur (are processed) after a certain amount of
(simulated) time has passed. They allow a process to sleep (or hold its
state) for the given time.
A [**Timeout**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Timeout) and
all other events can be created by calling the appropriate method of
the [**Environment**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.core.html#simpy.core.Environment) that
the process lives in
([**Environment.timeout()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.core.html#simpy.core.Environment.timeout) for
example).

# Process Interaction

The [**Process**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Process) instance
that is returned
by [**Environment.process()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.core.html#simpy.core.Environment.process) can
be utilized for process interactions. The two most common examples of
this are to wait for another process to finish and to interrupt another
process while it is waiting for an event.

## Waiting for a Process

As it happens, a
SimPy [**Process**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Process) can
be used like an event (technically, a process actually *is* an event).
If you yield it, you are resumed once the process has finished. Imagine
a car-wash simulation where cars enter the car-wash and wait for the
washing process to finish. Or an airport simulation where passengers
must wait until a security check finishes.

Let\'s assume that the car from our last example magically became an
electric vehicle. Electric vehicles usually take a lot of time charging
their batteries after a trip. They must wait until their battery is
charged before they can start driving again.

We can model this with an additional charge() process for our car.
Therefore, we refactor our car to be a class with two process
methods: run() (which is the original car() process function)
and charge().

The run process is automatically started when Car is instantiated. A
new charge process is started every time the vehicle starts parking. By
yielding
the [**Process**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Process) instance
that [**Environment.process()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.core.html#simpy.core.Environment.process) returns,
the run process starts waiting for it to finish.

# Interrupting Another Process

Imagine, you don't want to wait until your electric vehicle is fully
charged but want to interrupt the charging process and just start
driving instead.

SimPy allows you to interrupt a running process by calling
its [**interrupt()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html#simpy.events.Process.interrupt) method.

In Example3, the driver process has a reference to the
car's action process. After waiting for 3 time steps, it interrupts that
process.

Interrupts are thrown into process functions
as [**Interrupt**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.exceptions.html#simpy.exceptions.Interrupt) exceptions
that can (should) be handled by the interrupted process. The process can
then decide what to do next (e.g., continuing to wait for the original
event or yielding a new event).

# Shared Resources

SimPy offers three types
of [**resources**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#module-simpy.resources) that
help you modeling problems, where multiple processes want to use a
resource of limited capacity (e.g., cars at a fuel station with a
limited number of fuel pumps) or classical producer-consumer problems.

## Basic Resource Usage

We'll slightly modify our electric vehicle process car that we
introduced in the last sections.

The car will now drive to a *battery charging station (BCS)* and request
one of its two *charging spots*. If both spots are currently in use, it
waits until one of them becomes available again. It then starts charging
its battery and leaves the station afterwards.

The
resource's [**request()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource.request) method
generates an event that lets you wait until the resource becomes
available again. If you are resumed, you "own" the resource until
you *release* it.

If you use the resource with the with statement as in example 4, the
resource is automatically being released. If you
call request() without with, you are responsible to
call [**release()**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource.release) once
you are done using the resource.

When you release a resource, the next waiting process is resumed and now
"owns" one of the resource's slots. The
basic [**Resource**](https://simpy.readthedocs.io/en/latest/api_reference/simpy.resources.html#simpy.resources.resource.Resource) sorts
waiting processes in a *FIFO (first in---first out)* way.

# Appendix

## General Python 

### Generators

Taken from <https://www.geeksforgeeks.org/generators-in-python/>

A Generator in Python is a function that returns an iterator using the
Yield keyword. In this article, we will discuss how the generator
function works in Python.

A generator function in Python is defined like a normal function, but
whenever it needs to generate a value, it does so with the [yield
keyword](https://www.geeksforgeeks.org/python-yield-keyword/) rather
than return. If the body of a def contains yield, the function
automatically becomes a Python generator function. 

Create a Generator in Python

In Python, we can create a generator function by simply using the def
keyword and the yield keyword. The generator has the following syntax
in [Python](https://www.geeksforgeeks.org/python-programming-language/):

def function_name():\
yield statement

#### Generator Object

Python Generator functions return a generator object that is iterable,
i.e., can be used as
an [Iterator](https://www.geeksforgeeks.org/iterators-in-python/).
Generator objects are used either by calling the next method of the
generator object or using the generator object in a "for in" loop.

#### Python Generator Expression

In Python, generator expression is another way of writing the generator
function. It uses the Python [list
comprehension](https://www.geeksforgeeks.org/python-list-comprehension/) technique
but instead of storing the elements in a list in memory, it creates
generator objects.

Generator Expression Syntax

The generator expression in Python has the following Syntax:

(expression for item in iterable)

#### Applications of Generators in Python 

Suppose we create a stream of Fibonacci numbers, adopting the generator
approach makes it trivial; we just have to call next(x) to get the next
Fibonacci number without bothering about where or when the stream of
numbers ends. A more practical type of stream processing is handling
large data files such as log files. Generators provide a space-efficient
method for such data processing as only parts of the file are handled at
one given point in time. We can also use Iterators for these purposes,
but Generator provides a quick way (We don't need to write \_\_next\_\_
and \_\_iter\_\_ methods here).

#### When to use yield instead of return in Python?

The yield statement suspends a function's execution and sends a value
back to the caller, but retains enough state to enable the function to
resume where it left off. When the function resumes, it continues
execution immediately after the last yield run. This allows its code to
produce a series of values over time, rather than computing them at once
and sending them back like a list.

**Return** sends a specified value back to its caller
whereas **Yield** can produce a sequence of values. We should use yield
when we want to iterate over a sequence, but don't want to store the
entire sequence in memory. Yield is used in Python **generators**. A
generator function is defined just like a normal function, but whenever
it needs to generate a value, it does so with the yield keyword rather
than return. If the body of a def contains yield, the function
automatically becomes a generator function. 

Example:

+------------------------------------------------+
| \# A Python program to generate squares from 1 |
|                                                |
| \# to 100 using yield and therefore generator  |
|                                                |
|                                                |
|                                                |
| \# An infinite generator function that prints  |
|                                                |
| \# next square number. It starts with 1        |
|                                                |
|                                                |
|                                                |
|                                                |
|                                                |
| **def** nextSquare():                          |
|                                                |
|     i **=** 1                                  |
|                                                |
|                                                |
|                                                |
|     \# An Infinite loop to generate squares    |
|                                                |
|     **while** True:                            |
|                                                |
|         **yield** i**\***i                     |
|                                                |
|         i **+=** 1  \# Next execution resumes  |
|                                                |
|         \# from this point                     |
|                                                |
|                                                |
|                                                |
|                                                |
|                                                |
| \# Driver code to test above generator         |
|                                                |
| \# function                                    |
|                                                |
| **for** num **in** nextSquare():               |
|                                                |
|     **if** num \> 100:                         |
|                                                |
|         **break**                              |
|                                                |
|     **print**(num)                             |
+------------------------------------------------+

**Output:**

1

4

9

16

25

36

49

64

81

100

### Another example: <https://www.youtube.com/watch?v=KoH6FgVjnmg&t=481s>

### Callback functions in Python

Taken from:
<https://www.askpython.com/python/built-in-methods/callback-functions-in-python>

### A callback is a general concept in Python as well as other languages like Javascript, C, etc. We know that Python is an object-oriented language and functions are first-class objects in Python. This means, in Python, we can assign the value returned by a function to a variable and return a function from another function.

When one function is called from another function it is known as a
callback. A callback function is a function that is passed to another
function as an argument. It can be done in two ways:

1.  Passing one function as an argument to another function

2.  Calling a function inside another function

The above two points can help you decide if a given function is a
callback function or not. It is not necessary that both the above
conditions be true for a function to be a callback function. Even if one
of the above two conditions is satisfied, the function is termed a
callback function.

###  
