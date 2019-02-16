import pandas as pd
import numpy as np
from metrics import EpochRecord
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, HoverTool
from bokeh.io import output_notebook, push_notebook
from bokeh import palettes

class History:
    """Holds a list of EpochRecords matched with dataset phase and epoch_num
    """    
    def __init__(self, metrics=[], viz_params=dict(to_viz=False), phases=None, verbose=0):
        self.record = EpochRecord(metrics)
        self.df = pd.DataFrame(columns=['phase', 'epoch'] + [metric.name for metric in metrics])
        self.viz_params = viz_params
        self.viz_handle = False
        self.phases = phases
        self.verbose = verbose
        
        if viz_params['to_viz']:
            self.init_viz()
            
    def to_df(self):
        if not self.viz_params['to_viz']:
            return self.df
        else:
            return self.source.to_df().merge(self.df, how='outer', on=['phase', 'epoch'])
            
    def init_viz(self):
        # split the scalars to viz from the nonscalars
        self.source = ColumnDataSource(self.df.drop(labels=[metric.name for metric in self.record.metrics if not metric.is_scalar], axis=1))
        self.source.remove('index')
        self.df = self.df.drop(labels=[metric.name for metric in self.record.metrics if metric.is_scalar], axis=1)
            
        if self.viz_params['bands']:
            sem = lambda x: x.std() / np.sqrt(x.size)
            df2 = df.y.rolling(window=100).agg({'y_mean': np.mean, 'y_std': np.std, 'y_sem': sem})
            df2 = df2.fillna(method='bfill')

    def viz(self):
        if not self.viz_params['to_viz']:
            return None

        views = {}
        for phase in self.phases:
            views[phase] = CDSView(source=self.source, filters=[GroupFilter(column_name='phase', group=phase)])
            
        plots = []
        
        # bokeh.plotting.markers()
        markers = ['circle', 'triangle', 'square', 'diamond', 'hex', 'cross', 'asterisk', 'inverted_triangle']
        colors = palettes.all_palettes['Colorblind'][8]
        # generate a plot for each metric
        for metric in self.source.data.keys():
            if metric not in ['epoch', 'phase']:
                hover = HoverTool()
                hover.tooltips = [('epoch', '@{epoch}{0i}'), ('value', '@{}{{0.0000f}}'.format(metric))]
                
                p = figure(tools='pan,box_zoom,tap,lasso_select,save,reset', 
                           plot_width=290, 
                           plot_height=290,
                           title=metric)
                p.tools.append(hover)
                for i, phase in enumerate(self.phases):
                    p.scatter(x='epoch', 
                             y=metric, 
                             source=self.source, 
                             view=views[phase], 
                             legend=phase,
                             marker=markers[i],
                             color=colors[i],
                             size=7,
                             alpha=0.85)
                p.legend.location = 'bottom_left'
                # p.legend.click_policy = 'hide'
                p.legend.visible = False
                plots.append(p)
        plots[-1].legend.visible = True
        
        grid = []
        cur_row = []
        for plot in plots:
            if len(cur_row) >= 3:
                grid.append(cur_row)
                cur_row = []
            cur_row.append(plot)
        if len(cur_row) > 0:
            grid.append(cur_row)
        return grid
        
    def save_record(self, phase, epoch):
        self.record.computed.update({'phase': phase, 'epoch': epoch})
        if self.viz_params['to_viz']:
            # Filter out nonscalars (could also iterate thru metric names and check is_scalar)
            source = {k: np.array([v]) for k, v in self.record.computed.items() if np.shape(v) == ()}
            
            # separate the scalars to viz
            for k in source.keys():
                if k not in ['phase', 'epoch']:
                    del self.record.computed[k]
            self.source.stream(source)
            
            if self.viz_handle:
                push_notebook(self.viz_handle)
        if self.verbose == 1:
            print(self.record.computed)
        self.df = self.df.append(self.record.computed, ignore_index=True)
                
    def load_state_dict(self, state):
        # self.record = dill.loads(state['record']) # needed b/c of lambda functions in Metrics
        self.df = pd.DataFrame.from_dict(state['df'])
        self.viz_params = state['viz_params']
        self.phases = state['phases']
        
        if self.viz_params['to_viz']:
            self.source = ColumnDataSource(pd.DataFrame.from_dict(state['source']))
            self.source.remove('index')

    def state_dict(self):
        state = {}
        # state['record'] = dill.dumps(self.record)
        state['df'] = self.df.to_dict()
        state['viz_params'] = self.viz_params
        state['phases'] = self.phases
        
        if self.viz_params['to_viz']:
            state['source'] = self.source.to_df().to_dict()
        
        return state
