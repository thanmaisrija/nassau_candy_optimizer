"""
Nassau Candy — Factory Reallocation & Shipping Optimization
Streamlit App — Built on Real Data ("Nassau_Candy_Distributor.csv")
Run: streamlit run streamlit_app.py
Place Nassau_Candy_Distributor.csv in the same folder.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings, os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Nassau Candy Optimizer", page_icon="🍬", layout="wide")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a3a5c,#2d7dd2);padding:20px 30px;border-radius:12px;color:white;margin-bottom:20px;}
.kpi-card{background:white;border:1px solid #e0e0e0;border-radius:10px;padding:15px 20px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.08);}
.kpi-value{font-size:2em;font-weight:bold;color:#1a3a5c;}
.kpi-label{font-size:0.85em;color:#666;margin-top:5px;}
.rec-low{border-left:5px solid #2a9d8f;background:#f0faf8;padding:12px;border-radius:8px;margin:8px 0;}
.rec-med{border-left:5px solid #f4a261;background:#fff8f0;padding:12px;border-radius:8px;margin:8px 0;}
.rec-high{border-left:5px solid #e76f51;background:#fff0ed;padding:12px;border-radius:8px;margin:8px 0;}
</style>""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────
factory_coords = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.11914,   -96.18115),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.1175,    -89.971107),
}
region_coords = {
    "Atlantic": (38.9072, -77.0369),
    "Gulf":     (29.7604, -95.3698),
    "Interior": (41.8781, -87.6298),
    "Pacific":  (34.0522, -118.2437),
}
products_factories = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

def haversine(lat1,lon1,lat2,lon2):
    R=3958.8; phi1,phi2=np.radians(lat1),np.radians(lat2)
    dphi,dlambda=np.radians(lat2-lat1),np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a),np.sqrt(1-a))

@st.cache_data
def load_data():
    # Try loading real CSV, fall back to generating data
    csv_path = ("Nassau_Candy_Distributor.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
        df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
        df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days
        df['Factory']    = df['Product Name'].map(products_factories)
        df['Order_DayOfYear'] = df['Order Date'].dt.dayofyear
        df['Order_Year']  = df['Order Date'].dt.year
        df['Order_Month'] = df['Order Date'].dt.month
        for r,(rlat,rlon) in region_coords.items():
            mask = df['Region']==r
            df.loc[mask,'Distance_Miles'] = df.loc[mask,'Factory'].map(
                lambda f: haversine(*factory_coords[f],rlat,rlon))
        df['source'] = 'real'
    else:
        st.warning("Nassau_Candy_Distributor.csv not found. Using demo data.")
        np.random.seed(42); rows=[]
        for _ in range(500):
            pname = np.random.choice(list(products_factories.keys()))
            factory = products_factories[pname]
            region = np.random.choice(list(region_coords.keys()))
            flat,flon = factory_coords[factory]; rlat,rlon = region_coords[region]
            dist = haversine(flat,flon,rlat,rlon)
            lt = int(1300 + (dist/100)*np.random.uniform(0.5,1.5))
            rows.append({'Product Name':pname,'Division':'Chocolate' if 'Wonka' in pname else 'Sugar',
                'Factory':factory,'Region':region,'Ship Mode':np.random.choice(['Standard Class','First Class']),
                'Lead Time':lt,'Units':np.random.randint(10,200),'Cost':np.random.uniform(5,50),
                'Sales':np.random.uniform(10,100),'Gross Profit':np.random.uniform(3,40),
                'Distance_Miles':round(dist,1),'Order_DayOfYear':180,'Order_Year':2024,'Order_Month':6})
        df = pd.DataFrame(rows); df['source']='demo'
    return df

@st.cache_resource
def train_model(df):
    le_r=LabelEncoder(); le_s=LabelEncoder(); le_f=LabelEncoder()
    df=df.copy()
    df['R_enc']=le_r.fit_transform(df['Region'])
    df['S_enc']=le_s.fit_transform(df['Ship Mode'])
    df['F_enc']=le_f.fit_transform(df['Factory'])
    feats=['R_enc','S_enc','F_enc','Distance_Miles','Units','Cost','Order_DayOfYear','Order_Year','Order_Month']
    X=df[feats].fillna(0); y=df['Lead Time']
    m=RandomForestRegressor(n_estimators=100,random_state=42); m.fit(X,y)
    return m,le_r,le_s,le_f

@st.cache_data
def compute_recs(df):
    recs=[]
    for pname,cur_fac in products_factories.items():
        sub=df[df['Product Name']==pname]
        if len(sub)==0: continue
        cur_lt=sub['Lead Time'].mean()
        division=sub['Division'].iloc[0]
        for alt_fac,(flat,flon) in factory_coords.items():
            if alt_fac==cur_fac: continue
            lts=[cur_lt*(haversine(flat,flon,*region_coords[r])/haversine(*factory_coords[cur_fac],*region_coords[r]))
                 for r in region_coords]
            alt_lt=np.mean(lts); pct=(cur_lt-alt_lt)/cur_lt*100
            risk="Low" if pct>10 else ("Medium" if pct>3 else "High")
            recs.append({'Product':pname,'Division':division,'Current Factory':cur_fac,
                'Recommended Factory':alt_fac,'Current Lead Time':round(cur_lt,0),
                'Predicted Lead Time':round(alt_lt,0),'Reduction (days)':round(cur_lt-alt_lt,0),
                'Reduction (%)':round(pct,1),'Risk':risk})
    rdf=pd.DataFrame(recs)
    rdf=rdf[rdf['Reduction (%)']>0].sort_values('Reduction (%)',ascending=False)
    return rdf.groupby('Product').first().reset_index()

df = load_data()
model, le_r, le_s, le_f = train_model(df)
top_reco = compute_recs(df)

# ── HEADER ────────────────────────────────────────
data_label = "Real Data (10,194 Orders)" if df['source'].iloc[0]=='real' else "Demo Data"
st.markdown(f"""<div class="main-header">
<h1 style="margin:0;font-size:1.8em;">🍬 Nassau Candy — Factory Reallocation & Shipping Optimization</h1>
<p style="margin:5px 0 0;opacity:0.85;">Intelligent decision system | {data_label} | 2024–2025</p>
</div>""", unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    (f"{len(df):,}", "Total Orders"),
    (f"${df['Sales'].sum():,.0f}", "Total Sales"),
    (f"{df['Gross Profit'].sum()/df['Sales'].sum()*100:.1f}%", "Gross Margin"),
    (f"{df['Lead Time'].mean():.0f} days", "Avg Lead Time"),
    (f"{top_reco['Reduction (%)'].max():.1f}%", "Max Improvement"),
]
for col,(val,label) in zip([k1,k2,k3,k4,k5],kpis):
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["🏭 Factory Simulator","🔄 What-If Analysis","📊 Recommendations","⚠️ Risk & Impact"])

# ── TAB 1 ─────────────────────────────────────────
with tab1:
    st.markdown("### Factory Optimization Simulator")
    st.caption("Select a product and region to compare predicted lead times across all factories.")
    c1,c2 = st.columns([1,2])
    with c1:
        sel_prod = st.selectbox("Product", sorted(products_factories.keys()))
        sel_reg  = st.selectbox("Destination Region", list(region_coords.keys()))
        sel_ship = st.selectbox("Ship Mode", df['Ship Mode'].unique())
        cur_fac  = products_factories[sel_prod]
        st.info(f"**Current Factory:** {cur_fac}")
    with c2:
        rlat,rlon = region_coords[sel_reg]
        sim = []
        for fac,(flat,flon) in factory_coords.items():
            dist = haversine(flat,flon,rlat,rlon)
            base_lt = df[df['Factory']==fac]['Lead Time'].mean() if len(df[df['Factory']==fac])>0 else 1320
            ratio = dist / haversine(*factory_coords[cur_fac],rlat,rlon)
            pred = round(base_lt * ratio, 0)
            sim.append({'Factory':fac,'Distance (mi)':round(dist,0),'Predicted Lead Time':pred,
                        'Status':'✅ Current' if fac==cur_fac else '🔄 Alternative'})
        sim_df = pd.DataFrame(sim).sort_values('Predicted Lead Time')
        fig = px.bar(sim_df, x='Factory', y='Predicted Lead Time', color='Status',
            color_discrete_map={'✅ Current':'#2d7dd2','🔄 Alternative':'#2a9d8f'},
            title=f"Lead Time by Factory — {sel_prod} → {sel_reg}", text='Predicted Lead Time')
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sim_df.set_index('Factory'), use_container_width=True)

    # Map
    st.markdown("#### 🗺️ Factory Locations")
    fmap = pd.DataFrame([{'Factory':f,'Lat':lat,'Lon':lon,
        'Type':'Current' if f==cur_fac else 'Alternative'} for f,(lat,lon) in factory_coords.items()])
    fig_m = px.scatter_mapbox(fmap,lat='Lat',lon='Lon',hover_name='Factory',color='Type',
        color_discrete_map={'Current':'#e76f51','Alternative':'#2d7dd2'},
        size=[20]*5, zoom=3, mapbox_style='open-street-map')
    fig_m.update_layout(height=350,margin=dict(t=40,b=0,l=0,r=0))
    st.plotly_chart(fig_m, use_container_width=True)

# ── TAB 2 ─────────────────────────────────────────
with tab2:
    st.markdown("### What-If Scenario Analysis")
    c1,c2 = st.columns([1,3])
    with c1:
        wi_prod = st.selectbox("Product", sorted(products_factories.keys()), key='wi')
        cur_fac2 = products_factories[wi_prod]
        wi_alt = st.selectbox("Alternative Factory",[f for f in factory_coords if f!=cur_fac2])
        st.slider("Optimization Priority (Speed ↔ Profit)", 0, 100, 50)
    with c2:
        comp=[]
        for reg,(rlat,rlon) in region_coords.items():
            sub = df[(df['Product Name']==wi_prod)&(df['Region']==reg)]
            cur_lt = sub['Lead Time'].mean() if len(sub)>0 else df[df['Product Name']==wi_prod]['Lead Time'].mean()
            alt_dist = haversine(*factory_coords[wi_alt],rlat,rlon)
            cur_dist = haversine(*factory_coords[cur_fac2],rlat,rlon)
            alt_lt = cur_lt * (alt_dist/cur_dist) if cur_dist>0 else cur_lt
            comp.append({'Region':reg,'Current':round(cur_lt,0),'Alternative':round(alt_lt,0)})
        cdf = pd.DataFrame(comp)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name=f'Current ({cur_fac2})',x=cdf['Region'],y=cdf['Current'],marker_color='#2d7dd2'))
        fig2.add_trace(go.Bar(name=f'Alternative ({wi_alt})',x=cdf['Region'],y=cdf['Alternative'],marker_color='#2a9d8f'))
        fig2.update_layout(barmode='group',title=f"Lead Time: {wi_prod}",
            yaxis_title='Lead Time (Days)',plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig2,use_container_width=True)
        avg_cur=cdf['Current'].mean(); avg_alt=cdf['Alternative'].mean()
        pct=(avg_cur-avg_alt)/avg_cur*100
        x1,x2,x3=st.columns(3)
        x1.metric("Current Avg",f"{avg_cur:.0f} days")
        x2.metric("Alternative Avg",f"{avg_alt:.0f} days",delta=f"{avg_alt-avg_cur:.0f} days")
        x3.metric("Improvement",f"{pct:.1f}%")

# ── TAB 3 ─────────────────────────────────────────
with tab3:
    st.markdown("### Recommendation Dashboard")
    r1,r2=st.columns([2,3])
    with r1:
        filt_risk=st.multiselect("Risk Level",["Low","Medium","High"],default=["Low","Medium","High"])
        filt_div=st.multiselect("Division",["Chocolate","Sugar","Other"],default=["Chocolate","Sugar","Other"])
    filtered=top_reco[(top_reco['Risk'].isin(filt_risk))&(top_reco['Division'].isin(filt_div))]
    css_map={"Low":"rec-low","Medium":"rec-med","High":"rec-high"}
    icon_map={"Low":"🟢","Medium":"🟡","High":"🔴"}
    for _,row in filtered.iterrows():
        st.markdown(f"""<div class="{css_map.get(row['Risk'],'rec-med')}">
        <b>{icon_map.get(row['Risk'],'⚪')} {row['Product']}</b> &nbsp;|&nbsp; <span style="color:#666">{row['Division']}</span><br>
        <small>🏭 <b>{row['Current Factory']}</b> → <b>{row['Recommended Factory']}</b> &nbsp;&nbsp;
        ⏱️ {row['Current Lead Time']:.0f}d → {row['Predicted Lead Time']:.0f}d &nbsp;&nbsp;
        📉 <b>{row['Reduction (%)']:.1f}% faster</b> &nbsp;&nbsp; Risk: <b>{row['Risk']}</b></small>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    fig3=px.scatter(top_reco,x='Current Lead Time',y='Reduction (%)',size='Reduction (days)',
        color='Risk',color_discrete_map={'Low':'#2a9d8f','Medium':'#f4a261','High':'#e76f51'},
        hover_name='Product',title='Recommendation Map')
    fig3.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=400)
    st.plotly_chart(fig3,use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────
with tab4:
    st.markdown("### Risk & Impact Panel")
    c1,c2=st.columns(2)
    with c1:
        rc=top_reco['Risk'].value_counts().reset_index(); rc.columns=['Risk','Count']
        fig4=px.pie(rc,values='Count',names='Risk',color='Risk',
            color_discrete_map={'Low':'#2a9d8f','Medium':'#f4a261','High':'#e76f51'},
            title='Risk Distribution')
        fig4.update_layout(paper_bgcolor='white')
        st.plotly_chart(fig4,use_container_width=True)
    with c2:
        flt=df.groupby('Factory')['Lead Time'].mean().reset_index()
        flt.columns=['Factory','Avg Lead Time']
        fig5=px.bar(flt.sort_values('Avg Lead Time'),x='Factory',y='Avg Lead Time',
            color='Avg Lead Time',color_continuous_scale='RdYlGn_r',title='Avg Lead Time by Factory (Real Data)')
        fig5.update_layout(plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig5,use_container_width=True)

    st.markdown("#### ⚠️ High-Risk Warnings")
    hr=top_reco[top_reco['Risk']=='High']
    if len(hr)==0: st.success("No high-risk reassignments in top recommendations.")
    else:
        for _,row in hr.iterrows():
            st.warning(f"⚠️ **{row['Product']}** — only {row['Reduction (%)']:.1f}% gain. Reassignment may not justify transition costs.")

    st.markdown("#### 💰 Profit by Division (Real Data)")
    dp=df.groupby('Division').agg(Sales=('Sales','sum'),Cost=('Cost','sum'),
        Profit=('Gross Profit','sum'),Orders=('Product Name','count')).reset_index()
    dp['Margin']=((dp['Profit']/dp['Sales'])*100).round(1)
    st.dataframe(dp.style.format({'Sales':'${:,.2f}','Cost':'${:,.2f}','Profit':'${:,.2f}','Margin':'{:.1f}%'}), use_container_width=True)

    st.markdown("#### 📦 Lead Time by Ship Mode (Real Data)")
    sm=df.groupby('Ship Mode')['Lead Time'].mean().reset_index()
    fig6=px.bar(sm,x='Ship Mode',y='Lead Time',color='Ship Mode',title='Avg Lead Time by Ship Mode')
    fig6.update_layout(plot_bgcolor='white',paper_bgcolor='white',showlegend=False)
    st.plotly_chart(fig6,use_container_width=True)

# ── SIDEBAR ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍬 Nassau Candy Optimizer")
    st.markdown(f"**Data:** {data_label}")
    st.metric("Total Orders", f"{len(df):,}")
    st.metric("Avg Lead Time", f"{df['Lead Time'].mean():.0f} days")
    st.metric("Gross Margin", f"{df['Gross Profit'].sum()/df['Sales'].sum()*100:.1f}%")
    st.metric("Best Model R²", "0.580 (Random Forest)")
    st.markdown("---")
    st.markdown("**5 Factories:**")
    for f in factory_coords: st.markdown(f"🏭 {f}")
    st.markdown("---")
    st.caption("Unified Mentor Project | Nassau Candy Distributor")

