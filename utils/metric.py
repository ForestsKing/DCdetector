from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events


def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P, R, F
