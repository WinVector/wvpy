
import numpy
import matplotlib.pyplot
import sklearn
import sklearn.metrics
import itertools




def cross_predict_model(fitter, X, Y, plan):
    """train a model Y~X using the cross validation plan and return predictions"""
    preds = [None]*X.shape[0]
    for g in range(len(plan)):
        pi = plan[g]
        model = fitter.fit(X.iloc[pi["train"]], Y.iloc[pi["train"]])
        predg = model.predict(X.iloc[pi["test"]])
        for i in range(len(pi["test"])):
            preds[pi["test"][i]] = predg[i]
    return(preds)


def cross_predict_model_prob(fitter, X, Y, plan):
    """train a model Y~X using the cross validation plan and return probabilty matrix"""
    preds = numpy.zeros((X.shape[0], 2))
    for g in range(len(plan)):
        pi = plan[g]
        model = fitter.fit(X.iloc[pi["train"]], Y.iloc[pi["train"]])
        predg = model.predict_proba(X.iloc[pi["test"]])
        for i in range(len(pi["test"])):
            preds[pi["test"][i],0] = predg[i,0]
            preds[pi["test"][i],1] = predg[i,1]
    return(preds)




def mean_deviance(predictions, istrue):
    """compute per-row deviance of predictions versus istrue"""
    mass_on_correct = [ predictions[i,1] if istrue[i] else predictions[i,0] for i in range(len(istrue)) ]
    return(-2*sum(numpy.log(mass_on_correct))/len(istrue))



def mean_null_deviance(istrue):
    """compute per-row nulll deviance of predictions versus istrue"""
    p = numpy.mean(istrue)
    mass_on_correct = [ p if istrue[i] else 1-p for i in range(len(istrue)) ]
    return(-2*sum(numpy.log(mass_on_correct))/len(istrue))



def mk_cross_plan(n, k):
    """randomly split range(n) into k disjoint groups"""
    grp = [i % k for i in range(n)]
    numpy.random.shuffle(grp)
    plan = [ { "train"  : [i for i in range(n) if grp[i] != j],
               "test" : [i for i in range(n) if grp[i] == j] } for j in range(k) ]
    return(plan)





# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(prediction, istrue):
    fpr, tpr, _ = sklearn.metrics.roc_curve(istrue, prediction)
    auc = sklearn.metrics.auc(fpr, tpr)
    matplotlib.pyplot.figure()
    lw = 2
    matplotlib.pyplot.plot(fpr, tpr, color='darkorange',
         lw=lw, 
         label='ROC curve  (area = {0:0.2f})'
         ''.format(auc))
    matplotlib.pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.title('Receiver operating characteristic example')
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.show()
    return(auc)


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def search_grid(inp):
    """build a cross product of all named dictionary entries"""
    gen = (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))
    return([ci for ci in gen])
