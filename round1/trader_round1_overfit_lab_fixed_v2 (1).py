
from datamodel import OrderDepth, TradingState, Order
import jsonpickle

class Trader:
    LIMITS = {"INTARIAN_PEPPER_ROOT":80, "ASH_COATED_OSMIUM":80}
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # Change these fast and test
    CFG = {
        "OSM_MODE":"EMA",   # EMA / POLY / MOM
        "PEP_TARGET1":80, "PEP_TARGET2":80, "PEP_TARGET3":60, "PEP_TARGET4":20,
        "PEP_EDGE1":5.0, "PEP_EDGE2":4.0, "PEP_EDGE3":3.0, "PEP_EDGE4":2.0,
        "OSM_SOFT":78, "OSM_SPREAD_MIN":14, "OSM_QUOTE_GATE":0.6, "OSM_TAKE_GATE":2.8, "OSM_TAKE_EDGE":1.5,
        "OSM_A1":0.10, "OSM_A2":0.03,
        "OSM_EMA":0.75, "OSM_IMB":0.80, "OSM_IMB_D":0.15, "OSM_Z":-0.55, "OSM_MICRO":0.20, "OSM_PULL":0.08, "OSM_INV":-0.05,
        "OSM_T1":0.0, "OSM_T2":0.0, "OSM_T3":0.0,
        "OSM_MOM":0.90, "OSM_CURV":0.60,
        "L1":10, "L2":7, "L3":4, "BIAS":5, "TAKE_MAX":20,
    }

    def run(self, state: TradingState):
        d = self.load(state)
        self.dayreset(state, d)
        r = {self.PEPPER:[], self.OSMIUM:[]}
        if self.PEPPER in state.order_depths:
            r[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER])
        if self.OSMIUM in state.order_depths:
            r[self.OSMIUM] = self.trade_osmium(state, state.order_depths[self.OSMIUM], d)
        return r, 0, jsonpickle.encode(d)

    def load(self, state):
        if getattr(state, "traderData", None):
            try:
                d = jsonpickle.decode(state.traderData)
                if isinstance(d, dict):
                    d.setdefault("last_ts", None)
                    d.setdefault("osm_s", None)
                    d.setdefault("osm_l", None)
                    d.setdefault("osm_imb", 0.0)
                    d.setdefault("osm_hist", [])
                    return d
            except: pass
        return {"last_ts":None, "osm_s":None, "osm_l":None, "osm_imb":0.0, "osm_hist":[]}

    def dayreset(self, state, d):
        if d["last_ts"] is not None and state.timestamp < d["last_ts"]:
            d["osm_s"] = None; d["osm_l"] = None; d["osm_imb"] = 0.0; d["osm_hist"] = []
        d["last_ts"] = state.timestamp

    def pos(self, state, p): return state.position.get(p, 0)
    def book(self, od): return sorted(od.buy_orders.items(), reverse=True), sorted(od.sell_orders.items())
    def bbo(self, od):
        b,a = self.book(od)
        bb = b[0][0] if b else None; ba = a[0][0] if a else None
        bv = b[0][1] if b else 0; av = -a[0][1] if a else 0
        return bb,ba,bv,av
    def mid(self, od):
        bb,ba,_,_ = self.bbo(od)
        if bb is not None and ba is not None: return (bb+ba)/2
        return float(bb if bb is not None else ba) if (bb is not None or ba is not None) else None
    def micro(self, od):
        bb,ba,bv,av = self.bbo(od)
        if bb is None or ba is None: return self.mid(od)
        t = bv+av
        return (ba*bv + bb*av)/t if t>0 else (bb+ba)/2
    def spread(self, od):
        bb,ba,_,_ = self.bbo(od)
        return None if bb is None or ba is None else ba-bb
    def imb(self, od):
        _,_,bv,av = self.bbo(od)
        d = bv+av
        return 0.0 if d<=0 else (bv-av)/d
    def ema(self, prev, x, a): return x if prev is None else a*x + (1-a)*prev
    def meanstd(self, xs):
        if not xs: return 0.0,1.0
        m = sum(xs)/len(xs); v = sum((x-m)*(x-m) for x in xs)/max(1,len(xs))
        s = v**0.5
        return m, s if s>1e-6 else 1e-6
    def addb(self, os,p,px,q,pos,lim):
        q=min(q, lim-pos)
        if q>0: os.append(Order(p,int(px),int(q))); pos += q
        return pos
    def adds(self, os,p,px,q,pos,lim):
        q=min(q, lim+pos)
        if q>0: os.append(Order(p,int(px),int(-q))); pos -= q
        return pos

    def pep_fv(self, state, od):
        m = self.mid(od); mp = self.micro(od); im = self.imb(od)
        if m is None: return 0.0
        anchor = 1000*round((m-0.001*state.timestamp)/1000.0)
        fv = anchor + 0.001*state.timestamp
        if mp is not None: fv += 0.25*(mp-m)
        fv += 0.80*im
        return fv

    def pep_target(self, t):
        if t < 200000: return self.CFG["PEP_TARGET1"]
        if t < 920000: return self.CFG["PEP_TARGET2"]
        if t < 980000: return self.CFG["PEP_TARGET3"]
        return self.CFG["PEP_TARGET4"]

    def trade_pepper(self, state, od):
        p=self.PEPPER; lim=self.LIMITS[p]; pos=self.pos(state,p)
        bb,ba,_,_=self.bbo(od); m=self.mid(od); sp=self.spread(od)
        if bb is None or ba is None or m is None: return []
        fv=self.pep_fv(state,od); tgt=self.pep_target(state.timestamp); gap=tgt-pos
        orders=[]; _,asks=self.book(od)
        if gap>0:
            t=state.timestamp
            edge = self.CFG["PEP_EDGE1"] if t<100000 else self.CFG["PEP_EDGE2"] if t<400000 else self.CFG["PEP_EDGE3"] if t<900000 else self.CFG["PEP_EDGE4"]
            if gap>=50: edge += 1.0
            rem=min(gap, lim-pos)
            for px,qsgn in asks:
                q=-qsgn
                if rem<=0: break
                if px <= fv + edge:
                    take=min(q, rem)
                    pos=self.addb(orders,p,px,take,pos,lim); rem-=take
                else: break
        gap=tgt-pos
        bid=min(bb + (1 if sp is not None and sp>=2 else 0), int(round(fv)))
        if bid>=ba: bid=ba-1
        buy=30 if gap>40 else 20 if gap>20 else 12 if gap>5 else 4
        rich=bb-fv
        if pos>tgt+10 or rich>=5:
            ask=max(bb+1, min(ba, int(round(fv+2)))); sell=10 if pos>tgt+20 else 5
        elif pos>=78 and sp is not None and sp>=8:
            ask=ba; sell=2
        else:
            ask=max(ba, int(round(fv+8))); sell=0
        pos=self.addb(orders,p,bid,buy,pos,lim)
        pos=self.adds(orders,p,ask,sell,pos,lim)
        return orders

    def poly(self, ts):
        x=(ts/1000000.0)*2.0-1.0
        return self.CFG["OSM_T1"]*x + self.CFG["OSM_T2"]*x*x + self.CFG["OSM_T3"]*x*x*x

    def trade_osmium(self, state, od, d):
        p=self.OSMIUM; lim=self.LIMITS[p]; soft=self.CFG["OSM_SOFT"]; pos=self.pos(state,p)
        bb,ba,_,_=self.bbo(od); m=self.mid(od); sp=self.spread(od); mp=self.micro(od); im=self.imb(od)
        if None in (bb,ba,m,sp): return []
        d["osm_s"]=self.ema(d["osm_s"], m, self.CFG["OSM_A1"]); d["osm_l"]=self.ema(d["osm_l"], m, self.CFG["OSM_A2"])
        hist=d["osm_hist"]; hist.append(m)
        if len(hist)>80: del hist[0]
        mu,sig=self.meanstd(hist[-30:] if len(hist)>=10 else hist); z=(m-mu)/sig if sig>1e-9 else 0.0
        imb_d=im - d["osm_imb"]; d["osm_imb"]=im
        fv=10000.0 + self.CFG["OSM_EMA"]*(d["osm_s"]-d["osm_l"]) + self.CFG["OSM_IMB"]*im + self.CFG["OSM_IMB_D"]*imb_d + self.CFG["OSM_Z"]*z + self.CFG["OSM_PULL"]*(10000.0-m) + self.CFG["OSM_INV"]*pos
        if mp is not None: fv += self.CFG["OSM_MICRO"]*(mp-m)
        if self.CFG["OSM_MODE"]=="POLY":
            fv += self.poly(state.timestamp)
        elif self.CFG["OSM_MODE"]=="MOM" and len(hist)>=6:
            mom=hist[-1]-hist[-4]
            curv=hist[-1]-3*hist[-2]+3*hist[-3]-hist[-4]
            fv += self.CFG["OSM_MOM"]*mom + self.CFG["OSM_CURV"]*curv
        sigv=fv-m; ab=abs(sigv)
        orders=[]; bids,asks=self.book(od)
        if sp >= self.CFG["OSM_SPREAD_MIN"] and ab >= self.CFG["OSM_TAKE_GATE"]:
            if sigv>0 and pos<soft:
                rem=soft-pos
                for px,qsgn in asks:
                    q=-qsgn
                    if px <= fv - self.CFG["OSM_TAKE_EDGE"]:
                        take=min(q, self.CFG["OSM_TAKE_MAX"], rem)
                        pos=self.addb(orders,p,px,take,pos,lim); rem-=take
                        if rem<=0: break
                    else: break
            elif sigv<0 and pos>-soft:
                rem=soft+pos
                for px,q in bids:
                    if px >= fv + self.CFG["OSM_TAKE_EDGE"]:
                        take=min(q, self.CFG["OSM_TAKE_MAX"], rem)
                        pos=self.adds(orders,p,px,take,pos,lim); rem-=take
                        if rem<=0: break
                    else: break
        if not self.CFG["OSM_NEUTRAL_QUOTE"]:
            if sp < self.CFG["OSM_SPREAD_MIN"] or ab < self.CFG["OSM_QUOTE_GATE"]:
                return orders
        hw=4 if sp>=24 else 3 if sp>=18 else 2
        ib=1 if sigv>0.6 and sp>=2 else 0
        ia=1 if sigv<-0.6 and sp>=2 else 0
        bid_lv=[min(bb+ib, int(round(fv-hw))), int(round(fv-hw-1)), int(round(fv-hw-2))]
        ask_lv=[max(ba-ia, int(round(fv+hw))), int(round(fv+hw+1)), int(round(fv+hw+2))]
        bs=[self.CFG["L1"], self.CFG["L2"], self.CFG["L3"]]
        ss=[self.CFG["L1"], self.CFG["L2"], self.CFG["L3"]]
        if sigv>1.0:
            bs=[x+self.CFG["BIAS"] for x in bs]; ss=[max(1,x-self.CFG["BIAS"]) for x in ss]
        elif sigv<-1.0:
            ss=[x+self.CFG["BIAS"] for x in ss]; bs=[max(1,x-self.CFG["BIAS"]) for x in bs]
        if pos>55:
            bs=[max(0,x-6) for x in bs]; ss=[x+4 for x in ss]
        elif pos<-55:
            ss=[max(0,x-6) for x in ss]; bs=[x+4 for x in bs]
        for px,q in zip(bid_lv,bs):
            if q>0 and px<ba: pos=self.addb(orders,p,px,q,pos,lim)
        for px,q in zip(ask_lv,ss):
            if q>0: pos=self.adds(orders,p,max(px,bb+1),q,pos,lim)
        return orders
