# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:49:37 2018

@author: victzuan
"""
#Built from: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
#Operations:
#Every operation is characterized by three things:
#
# 1- A compute function that computes the operation’s output given values for 
#   the operation’s inputs;
# 2- A list of input_nodes which can be variables or other operations;
# 3- A list of consumers that use the operation’s output as their input.

class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """
    
    def __init__(self, input_nodes=[]):
        '''Construct Operation.
    
        '''
        self.input_nodes = input_nodes
        
        #Initialize list of consumers ().
        #(i.e. nodes that receive this operation's output as input)
        self.consumers = []
        
        #Append this operation to the list of consumers of all input nodes.
        for input_node in input_nodes:
            input_node.consumers.append(self)
        
        #Append this operation to the list of operations in the currently 
        #active defaut graph.
        _defaut_graph.operations.append(self)
        
    def compute(self):
        """ Computes the output of this operation.
            Must be implemented by the particular operation.
        """
        pass

#Some elementary operations
#Let’s implement some elementary operations in order to become familiar with 
#the Operation class (and because we will need them later). In both of these 
#operations, we assume that the tensors are NumPy arrays, in which the 
#element-wise addition and matrix multiplication (.dot) are already implemented 
#for us.    
    
class add(Operation):
    ''' Returns x + y element wise.
    '''
    
    def __init__(self, x, y):
        '''Construct add
        
        Args:
            x: First summand node.
            y: Second summand node.
        '''
        super().__init__([x,y])
    
    def compute(self, x_value, y_value):
        '''Compute the output of the add operation.
        
        Args:
            x_value: First summand value.
            y_value: Second summand value.
        '''
        return x_value + y_value
    

class matmul(Operation):
    '''Multiplies matrix a by matrix b, producing a * b.
    '''
    def __init__(self,a,b):
        '''Construct matmul.
        
        Args:
            a: First matrix.
            b: Second matrix.
        '''
        
        super().__init__([a,b])
        
    
    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        return a_value.dot(b_value)
    
    
class sigmoid(Operation):
    '''Returns the sigmoid of x element-wise.'''
    
    def __init__(self, a):
        '''Construct sigmoid
        
        Args::
            a: input node
        '''
        super().__init__([a])
        
    def compute(self, a_value):
        '''Compute the output of the sigmoid operation.
        
        Args:
            a_value: Input value.
        Return:
            output of the sigmoid operation for a_value.
        '''
        return 1/(1+np.exp(-a_value))

#Softmax
#When the output classes are disjoint, the output probabilities should sum up 
#to one. In this case, we apply the softmax function to the vector a=xW+b 
#instead of the element-wise sigmoid, which makes sure that each probability 
#is between 0 and 1 and the sum of the probabilities is 1:
    
class softmax(Operation):
    '''Returns the softmax of a.
    '''
    
    def __init__(self, a):
        '''Construct softmax.
        
        Args:
            a: Input node
        '''    
        super().__init__([a])
    
    def compute(self, a_value):
        '''Computes the output of the softmax operation.
            Args:
                a_value: Input value
        '''
        return np.exp(a_value)/ np.sum(np.exp(a_value), axis = 1)[:, None]

class log(Operation):
    '''Computes the natural logarithm of x element-wise.
    '''
    def __init__(self, x):
        '''Construct log.
        
        Args:
            x: input node.
        '''
        super().__init__([x])
    
    def compute(self, x_value):
        '''Compute the output of the log operation.
        
        Args:
            x_value: Input value.
        '''
        return np.log(x_value)

class multiply(Operation):
    ''' Returns x*y element-wise.
    '''
    def __init__(self, x, y):
        '''Construct multiply.
        
        Args:
            x: first mutiplicand node.
            y: second multiplicand node.
        '''
        super().__init__([x, y])
    
    def compute(self, x_value, y_value):
        '''Compute the output of the multiply operation
        
        Args:
            x_value: First multiplicand value.
            y_value: Second multiplicand value.
        '''
        return x_value*y_value
    
class reduce_sum(Operation):
    '''Computes the sum of elements across dimensions of a tensor.
    '''
    def __init__(self, A, axis = None):
        '''Construct reduce_sum.
        
        Args:
            A: The tensor to reduce.
            axis: The dimensions to reduce. If 'None' (the default), reduces
                all dimensions.
        '''
        super().__init__([A])
        self.axis = axis
    
    def compute(self, A_value):
        '''Compute the output of the reduce_sum operation.
        Args:
            A_value: Input tensor value
        '''
        return np.sum(A_value, self.axis)

class negative(Operation):
    '''Computes the negatie of x element-wise.
    '''
    def __init__(self, x):
        '''Construct negative.
        Args:
            x: Input node.
        '''
        super().__init__([x])
    
    def compute(self, x_value):
        '''Compute the output of the negative operation.
        
        Args:
            x_value: Input value.
        '''
        return -x_value
  
#Placeholders:
#Not all the nodes in a computational graph are operations. For example, in 
#the affine transformation graph, A, x and b are not operations. Rather, they 
#are inputs to the graph that have to be supplied with a value once we want to 
#compute the output of the graph. To provide such values, we introduce 
#placeholders.
    
class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """
    
    def __init__(self):
        """Construct placeholder.
        """
        self.consumers = []
        
        #Append this placeholder to the list of placeholders in the currently 
        #active defaut graph.
        _defaut_graph.placeholders.append(self)

#Variables:
#In the affine transformation graph, there is a qualitative difference between
# x on the one hand and A and b on the other hand. While x is an input to the 
#operation, A and b are parameters of the operation, i.e. they are intrinsic 
#to the graph. We will refer to such parameters as Variables.
        
class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a 
        computational graph).
    """
    
    def __init__ (self, initial_value=None):
        """Construct Variable.

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []
        
        # Append this variable to the list of variables in the currently
        #active defaut graph.
        _defaut_graph.variables.append(self)
        
#The Graph class:
#Finally, we’ll need a class that bundles all the operations, placeholders and 
#variables together. When creating a new graph, we can call its as_default 
#method to set the _default_graph to this graph. This way, we can create 
#operations, placeholders and variables without having to pass in a reference 
#to the graph everytime.

class Graph:
    """Represents a computational graph
    """
    
    def __init__(self):
        '''Construct Graph'''
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def as_default(self):
        global _defaut_graph
        _defaut_graph = self
        

# Computing the output of an operation:
# Now that we are confident creating computational graphs, we can start to 
#think about how to compute the output of an operation.
#
# Let’s create a Session class that encapsulates an execution of an operation. 
# We would like to be able to create a session instance and call a run method 
#on this instance, passing the operation that we want to compute and a 
#dictionary containing values for the placeholders:

#session = Session()
#output = session.run(z, {
#    x: [1, 2]
#})
#
# In order to compute the function represented by an operation, we need to 
#apply the computations in the right order, such that the values of every node 
#that is an input to an operation o has been computed before o is computed. 
#This can be achieved via post-order traversal.
    
import numpy as np

class Session:
    ''' Represents a particular execution of a computational graph.
    '''
    def traverse_postorder(operation):
            """Performs a post-order traversal, returning a list of nodes
                in the order in which they have to be computed.
            
                Args:
                   operation: The operation to start traversal at
            """
            
            nodes_postorder = []
            
            def recurse(node):
                if isinstance(node, Operation):
                    for input_node in node.input_nodes:
                        recurse(input_node)
                nodes_postorder.append(node)
            
            recurse(operation)
            return nodes_postorder
        
    def run (self, operation, feed_dict = {}):
        """Computes the output of an operation.

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this 
              session.
        """
        
        # Perform a post-order traversal of the graph to bring the nodes into 
        #the right order.
        nodes_postorder = Session.traverse_postorder(operation)
        
        # Iterate all nodes to determine their value.
        for node in nodes_postorder:
            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict.
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # Set the node value to the variable's value attribute.
                node.output = node.value
            else: #Operation.
                # Get the input values for this operation from node_values.
                node.inputs = [input_node.output for input_node in node.input_nodes]
                
                #Compute the output of this operation.
                node.output = node.compute(*node.inputs)
            # Convert lists to numpy arrays.
            if type(node.output) == list:
                node.output = np.array(node.output)
            
        #Return the requested node value.
        return operation.output
    
from queue import Queue

#A dictionary that will map operations to get gradient functions.
_gradient_registry = {}

#As a prerequisite to implementing backpropagation, we need to specify a 
#function for each operation that computes the gradients with respect to the 
#inputs of that operation, given the gradients with respect to the output.  
class RegisterGradient:
    '''A decorator for registering the gradient function for an op type.
    '''
    def __init__(self, op_type):
        '''Creates a new decorator with 'op_type' as the Operation type.
        Args:
            op_type: The name of an operation.
        '''
        self._op_type = eval(op_type)
        
    def __call__(self, f):
        '''Registers the function 'f' as gradient function for 'op_type'.
        '''
        _gradient_registry[self._op_type] = f
        return f
#Gradient for negative.
#Given a gradient G with respect to −x, the gradient with respect to x is given 
#by −G.    
@RegisterGradient("negative")
def _negative_gradient(op, grad):
    '''Computes the gradients for 'negative'.
    
    Args:
        op: The 'negative' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'negative' op.
        
    Return:
        Gradients with respect to the input of 'negative'.
    '''
    return -grad

#Gradient for log.
#Given a gradient G with respect to log(x), the gradient with respect to x is 
#given by G/x.
@RegisterGradient("log")
def _log_gradient(op, grad):
    '''Computes the gradient for 'log'.
    
    Args:
        op: The 'log' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'log' op.
        
    Returns:
        Gradients with respect to the input of 'log'.
    '''
    x =  op.inputs[0]
    return grad/x

#Gradient for sigmoid
#Given a gradient G with respect to σ(a), the gradient with respect to a is 
#given by G⋅σ(a)⋅σ(1−a).
@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    '''Computes the gradients for 'sigmoid'.
    
    Args:
        op: The 'sigmoid' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'sigmoid' op.
        
    Returns:
        Gradients with respect to the input of 'sigmoid'.
    '''
    sigmoid = op.output
    return grad*sigmoid*(1-sigmoid)

#Gradient for multiply.
#Given a gradient G with respect to A⊙B, the gradient with respect to A is 
#given by G⊙B and the gradient with respect to B is given by G⊙A.
@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    '''Computes the gradiens for 'multiply'.
    
    Args:
        op: The 'multiply' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'multiply' op.
    
    Returns:
        Gradients with respect to the input of 'multiply'.
    '''
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad*B, grad*A]

#Gradient for matmul.
#Given a gradient G with respect to AB, the gradient with respect to A is 
#given by GBT and the gradient with respect to B is given by ATG.
@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    '''Computes the gradients for 'matmul'.
    
    Args:
        op: The 'matmul' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'matmul' op.
    
    Returns:
        Gradients with respect to the input of 'matmul'.
    '''
    
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T), A.T.dot(grad)]

#Gradient for add.
#Given a gradient G with respect to a+b, the gradient with respect to a is 
#given by G and the gradient with respect to b is also given by G, provided 
#that a and b are of the same shape. 
#If a and b are of different shapes, we assume that b is added to each row of a. 
@RegisterGradient("add")
def _add_gradient(op, grad):
    '''Computes the gradients for 'add'.
    
    Args:
        op: The 'add' 'Operation that we are differentiating.
        grad: Gradient with respect to the output of the 'add' op.
        
    Returns:
        Gradients with respect to the input of 'add'.
    '''
    a = op.inputs[0]
    b = op.inputs[1]
    
    grad_wrt_a = grad
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis = 0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis = axis, keepdims = True)
    
    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis = 0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis = axis, keepdims = True)
    
    return [grad_wrt_a, grad_wrt_b]

#Gradient for reduce_sum
#Given a gradient G with respect to the output of reduce_sum, the gradient 
#with respect to the input A is given by repeating G along the specified axis.
@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    '''Cmputes the gradients for 'reduce_sum'.
    
    Args:
        op: The 'reduce_sum' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'reduce_sum' op.
        
    Returns:
        Gradients with respect to the input of 'reduce_sum'.
    '''
    A = op.inputs[0]
    
    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape//output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)

#Gradient for softmax.
@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    '''Computes the gradients for 'softmax'.
    
    Args:
        op: The 'softmax' 'Operation' that we are differentiating.
        grad: Gradient with respect to the output of the 'softmax' op.
    
    Returns:
        Gradients with respect to the input of 'softmax'.
    '''
    softmax = op.output
    return (grad - np.reshape(
            np.sum(grad*softmax, 1),
            [-1,1]
            ))*softmax


#Assume that our _gradient_registry dictionary is already filled with gradient 
#computation functions for all of our operations. We can now implement 
#backpropagation.
def compute_gradients(loss):
    #grad_table[node] will contain the gradient of the loss w.r.t. the node's
    #output.
    grad_table = {}
    
    #The gradient of the loss with respect to the loss is just 1.
    grad_table[loss] = 1
    
    #Perform a breadt-first search, backwards from the loss.
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)
    
    while not queue.empty():
        node = queue.get()
        
        #If this node is not the loss:
        if node != loss:
            #compute the gradient of the loss with respect to this node's output.
            grad_table[node] = 0
            
            #Iterate all consumers:
            for consumer in node.consumers:
                #Retrive the gradient of the loss w.r.t. consumer's output.
                lossgrad_wrt_consumer_output = grad_table[consumer]
                
                #Retrive the function which computers gradients with respect 
                #to consumer's inputs given gradients with respect to consumer's
                #output.
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]
                
                #Get the gradient of the loss with respect to all of consumer's
                #inputs.
                lossgrads_wrt_consumer_inputs = bprop(
                        consumer, 
                        lossgrad_wrt_consumer_output
                        )
                if len(consumer.input_nodes) == 1:
                    #If there is a single input node to the consumer, 
                    #lossgrads_wrt_consumer_inputs is a scalar.
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    #Otherwise, lossgrads_wrt_consumer_inputs is an array of 
                    #gradients for each input node.
                    
                    #Retrive the index of node in consumer's inputs.
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    #Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    #Add to total gradient.
                    grad_table[node] += lossgrad_wrt_node
                    
        #Append each input node to the queue.
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    #Return gradients for each visited node.
    return grad_table
                        
    
    
#Let’s implement an operation that minimizes the value of a node using gradient
#descent. We require the user to specify the magnitude of the step along the 
#gradient as a parameter called learning_rate.
class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        #print ('I am at', self, 'learning rate: ', self.learning_rate)
    
    def minimizer(self, loss):
        learning_rate = self.learning_rate
        
        class MinimizationOperation(Operation):
            def compute(self):
                #Compute gradients.
                grad_table = compute_gradients(loss)
                
                #Iterate all variables.
                for node in grad_table:
                    if type(node) == Variable:
                        #Retrive gradient for this variable.
                        grad = grad_table[node]
                        #Take a step along the direction of the negative gradient
                        node.value -= learning_rate*grad
        return MinimizationOperation()




def example():       
    # Create a new graph
    Graph().as_default()
    
    # Create variables
    A = Variable([[1, 0], [0, -1]])
    b = Variable([1, 1])
    
    # Create placeholder
    x = placeholder()
    
    # Create hidden node y
    y = matmul(A, x)
    
    # Create output node z
    z = add(y, b)            
    
    session = Session()
    output = session.run(z, {
        x: [1, 2]
    })
    print(output)       
                           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    