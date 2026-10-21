import QuantLib as ql
import csv
import math
import re
from scipy.optimize import linprog
from datetime import datetime


## INPUTS -- SHARED

inputOptions = "/Users/adamfee/Desktop/FNCE 4820/ProjectTwo/spx 24Aug2022.csv"
inputExpiry = ql.Date(17, 3, 2023) # Day, month, year
inputTicker = "SPX"
inputPart = 2

## INPUTS -- PART 2 ONLY

riskFreeRate = 0.041
positionType = "put"
positionStrike = 6000
positionQuantity = -5.0 


## DATE FORMAT HELPERS

dateFormats = [
    "%B %d, %Y",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%d-%b-%Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%a %B %d %Y",
    "%A %B %d %Y",
    "%a %b %d %Y",
    "%a %b %d, %Y",
]

def parseDate(text):
    text = text.strip()
    for format in dateFormats:
        try:
            d = datetime.strptime(text, format).date()
            return ql.Date(d.day, d.month, d.year)
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {text!r}")


## CURRENT DATE HELPER (PART 2)

def today(path):
    with open(path, newline="") as file:
        line3 = list(csv.reader(file))[2][0]
    text = line3.split("Date:", 1)[-1]
    text = re.split(r"\d{1,2}:\d{2}", text)[0]
    text = text.strip().rstrip(",").replace(" at", "").strip()
    return parseDate(text)


## CURRENT POSITION PAYOFF HELPER (PART 2)

def positionPayoff(u):
    if positionType == "call":
        return positionQuantity * max(u - positionStrike, 0.0)
    elif positionType == "put":
        return positionQuantity * max(positionStrike - u, 0.0)
    else:
        raise ValueError(f"valid options are call/put {positionType}")


## OPTIONS INPUT WRITER

options = []
with open(inputOptions, newline="") as file:
    next(csv.reader(file))
    next(csv.reader(file))
    next(csv.reader(file))
    next(csv.reader(file))

    for row in csv.reader(file):
        if len(row) < 17 or float(row[5].strip()) <= 0 or float(row[16].strip()) <= 0:
            continue
        
        expirationDate = parseDate(row[0])
        ticker = row[1].strip()
        optionType = "SPXW" if ticker.startswith("SPXW") else "SPX"

        if optionType != inputTicker:
            continue

        if expirationDate != inputExpiry:
            continue

        options.append({
            "expirationDate": expirationDate,
            "callTicker": row[1].strip(),
            "callBid": float(row[4].strip()),
            "callAsk": float(row[5].strip()),
            "strike": float(row[11].strip()),
            "putTicker": row[12].strip(),
            "putBid": float(row[15].strip()),
            "putAsk": float(row[16].strip())
        })


## VALIDATIONS

if not options:
    raise ValueError(f"no options matched {inputExpiry} / {inputTicker} in {inputOptions}")

if positionStrike not in {option["strike"] for option in options}:
    raise ValueError(f"position strike {positionStrike} is not traded at this expiry")


## PART 1 MATRICES

def runPartOne(options):
    ## C MATRIX
    # LINPROG MINIMIZES, SO WE NEED TO NEGATE, WHEN SELLING A CALL AT THE BID -> MONEY COMES IN (NEGATED: OUTFLOW)

    n = len(options)
    c = [0.0] * (4 * n)

    for i, option in enumerate(options):
        c[0 * n + i] = -option["callBid"]
        c[1 * n + i] = option["callAsk"]
        c[2 * n + i] = -option["putBid"]
        c[3 * n + i] = option["putAsk"]


    ## A MATRIX
    # LINPROG MINIMIZES, SO WE NEED TO NEGATE, WHEN SELLING A CALL AT PAYOFF -> WE OWE MONEY (NEGATED: INFLOW)

    u = [0.0] + sorted ({option["strike"] for option in options})
    A_ub = []

    for j in u:
        row = [0.0] * (4 * n)

        for i, option in enumerate(options):
            k = option["strike"]
            call = max(j - k, 0)
            put = max(k - j, 0)
    
            row[0 * n + i] = call
            row[1 * n + i] = -call
            row[2 * n + i] = put
            row[3 * n + i] = -put

        A_ub.append(row)

    row = [0.0] * (4 * n)
    for i in range(n):
        row[0 * n + i] = 1.0
        row[1 * n + i] = -1.0

    A_ub.append(row)


    ## B MATRIX

    b = [0.0] * len(A_ub)

    bounds = [(0, 1)] * (n * 4)
    return linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")


## PART 2 MATRICES

def runPartTwo(options):
    n = len(options)

    ## C MATRIX

    c = [0.0] * (4 * n + 2)
    T = ql.Actual365Fixed().yearFraction(today(inputOptions), inputExpiry)
    discountFactor = math.exp(-riskFreeRate * T)

    for i, option in enumerate(options):
        c[0 * n + i] = -option["callBid"]
        c[1 * n + i] = option["callAsk"]
        c[2 * n + i] = -option["putBid"]
        c[3 * n + i] = option["putAsk"]

    c[4 * n] = -discountFactor
    c[4 * n + 1] = discountFactor

    ## A AND B MATRIX

    u = [0.0] + sorted ({option["strike"] for option in options})
    A_eq = []
    b_eq = []

    for j in u:
        row = [0.0] * (4 * n + 2)

        for i, option in enumerate(options):
            k = option["strike"]
            call = max(j - k, 0) # CALL PAYOFF
            put = max(k - j, 0) # PUT PAYOFF
    
            row[0 * n + i] = -call
            row[1 * n + i] = call
            row[2 * n + i] = -put
            row[3 * n + i] = put
        
        row[4 * n] = -1.0
        row[4 * n + 1] = 1.0

        A_eq.append(row)
        b_eq.append(-positionPayoff(j))
    
    row = [0.0] * (4 * n + 2)
    for i in range(n):
        row[0 * n + i] = -1.0
        row[1 * n + i] = 1.0

    A_eq.append(row)
    b_eq.append(-positionQuantity if positionType == "call" else 0.0)

    bondCap = max(option["strike"] for option in options) * abs(positionQuantity)
    bounds = [(0, None)] * (n * 4) + [(0, bondCap)] * 2
    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    
## PART CHECK AND SOLUTION

n = len(options)
tolerance = 1e-6

if inputPart == 1:
    res = runPartOne(options)
    value = -res.fun

    if res.status != 0:
        print("Solver failed:", res.message)
    elif value < tolerance:
        print(f"No arbitrage. ({n} strikes, {inputExpiry}, {inputTicker})")
    else:
        print(f"ARBITRAGE: {value:,.2f} per unit")
        labels = (["Sell call"] * n + ["Buy call"] * n + ["Sell put"] * n + ["Buy put"] * n)

        for i, quantity in enumerate(res.x):
            if abs(quantity) > tolerance:
                strike = options[i % n]["strike"]
                print(f"    {labels[i]:10s} K={strike:8.1f} {quantity:10.4f}")
elif inputPart == 2:
    res = runPartTwo(options)
    bondCap = max(option["strike"] for option in options) * abs(positionQuantity)

    if res.status != 0:
        print ("Solver failed:", res.message)
    else:
        print(f"Exit {positionQuantity:+g} {positionType} K = {positionStrike:g}")
        print(f"Net cash from exit: {-res.fun:,.2f}")

        for option in options:
            if positionStrike == option["strike"]:
                if positionType == "call":
                    naive = option["callBid"] if positionQuantity > 0 else -option["callAsk"]
                else:
                    naive = option["putBid"] if positionQuantity > 0 else -option["putAsk"]
                print(f"Naive exit: {naive * abs(positionQuantity):,.2f}")

        labels = (["Sell call"] * n + ["Buy call"] * n + ["Sell put"] * n + ["Buy put"] * n + ["Sell bond", "Buy bond"])

        for i, quantity in enumerate(res.x):
            if abs(quantity) > tolerance:
                if i < 4 * n:
                    print(f"    {labels[i]:10s} K ={options[i % n]['strike']:8.1f} {quantity:10.4f}")
                else:
                    print(f"    {labels[i]:10s} {'':11s}{quantity:10.4f}")
                    if labels[i] == "Buy bond" or labels[i] == "Sell bond":
                        if quantity > bondCap - tolerance:
                            print("WARNING: INPUT RATE OF", riskFreeRate, "DOESN'T MATCH OPTION QUOTES AND BONDS ARE PINNED AT CAP")
else:
    print("Wrong input")