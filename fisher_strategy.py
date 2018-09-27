# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:11:51 2018

@author: gabri
"""



import numpy as np
import pandas as pd

import plot_functions as pf



# get fisher transform. old price starts.
def get_fisher_transform_series(df_data, ts_high, ts_low, lookback, px_weight = 0.33, fisher_weight = 0.5, return_fisher_only = True):
    df = df_data[[ts_high, ts_low]].copy()
    df['px'] = (df[ts_high] + df[ts_low]) / 2.0    
    df['roll_max'] = df['px'].rolling(window=lookback).max()
    df['roll_min'] = df['px'].rolling(window=lookback).min()
    
    df['z'] = 2 * (df.px - df.roll_min) / (df.roll_max - df.roll_min) - 1.0
    df['z_ema'] = df.z.ewm(alpha = px_weight, adjust = False, ignore_na = True).mean()
    df.z_ema[df.z_ema > 0.99] = 0.999
    df.z_ema[df.z_ema < -0.99] = -0.99

    df['z_trans'] = np.log((1.0 + df.z_ema) / (1.0 - df.z_ema))
    df['fisher'] = df.z_trans.ewm(alpha = fisher_weight, adjust = False, ignore_na = True).mean()
    
    return df[['fisher']] if return_fisher_only else df



def backtest_single(df_data, ts_high, ts_low, ts_close, lot=1, fisher_lookback_days=10, fisher_ma_days=2, 
                 entry_lvl=2.0, exit_lvl=0.0, stop_trade_pnl=np.inf, stop_drawdown=np.inf, 
                 notional = 100000.0, print_results = False):
    
    # output history dataframe
    df_backtest = pd.DataFrame(index=df_data.index, 
                               columns=[ 'const', 'slope', 'sigma', 'e', 'z', 
                                        'pos_y','pos_x','px_y','px_x','trade_ratio','daily_pnl','daily_return','cum_pnl',
                                        'drawdown','drawdown_duration', 'new_trade_code'])
    df_backtest[[y_name,x_name]] = np.log(df_data[[y_name,x_name]]) if log_transform else df_data[[y_name,x_name]]
    df_backtest[['px_y','px_x']] = df_data[[y_name,x_name]]    
    df_backtest[['pos_y','pos_x','daily_pnl', 'new_trade_code']] = 0.0
    df_backtest['cum_pnl'] = notional
    df_backtest.dropna(subset = ['px_y','px_x'], inplace = True)
    df_trade = pd.DataFrame(columns=['entry_time','exit_time','exit_code','pos_y','pos_x','entry_px_y','entry_px_x',
                                     'exit_px_y','exit_px_x','entry_notional','entry_value','signal_ratio','trade_ratio','pnl','duration'])

    # initiation
    cum_pnl= notional
    trade_pnl = 0.0
    itrade = -1
    out_pos = False
    allow_enter = True
    stop_pos_y = 0.0
    kalman_model = kalman_strategy(delta = 0.0001)
    
    # test strategy at each time step
    for i in range(1,len(df_backtest.index)):
    
        signal = dict()
        y = df_backtest[y_name][i]
        x = df_backtest[x_name][i]
        
        if strategy_type == 'ols':
            # calculate OLS signal
            if i > ols_lookback_days:        
                y_train = df_backtest[y_name].iloc[i-ols_lookback_days:i]
                x_train = df_backtest[x_name].iloc[i-ols_lookback_days:i]                
                signal = get_ols_signal(x_train, y_train, x, y)
            
        elif strategy_type == 'kalman':
            # calculate Kalman filter signal
            y_train = df_backtest[y_name].iloc[i-1]
            x_train = df_backtest[x_name].iloc[i-1]
            kalman_model.update_filter(x_train, y_train)        
            if i > kalman_run_days: signal = kalman_model.get_signal(x, y)
        
        # set signals
        if not out_pos:
            const = signal.get('const')
            slope = signal.get('slope',0)
            sigma = signal.get('sigma')
            e = signal.get('e', 0)
            z = signal.get('z', 0)
        else:
            # use previous const and slope for exit condition
            e = y - const - slope * x
            z = e  / sigma
            
        # pnl
        pos_y = df_backtest['pos_y'].iloc[i-1]
        pos_x = df_backtest['pos_x'].iloc[i-1]
        px_y = df_backtest['px_y'].iloc[i]
        px_x = df_backtest['px_x'].iloc[i]
        trade_ratio = np.nan if pos_y ==0 else np.abs(pos_x*px_x / (pos_y*px_y) if log_transform else pos_x / pos_y)
                
        daily_pnl = np.sum((df_backtest[['px_y','px_x']].iloc[i] - df_backtest[['px_y','px_x']].iloc[i-1]) * [pos_y, pos_x])
        daily_return = daily_pnl / cum_pnl
        cum_pnl += daily_pnl
        trade_pnl += daily_pnl
        max_cum_pnl = np.max(df_backtest['cum_pnl'][0:i])
        drawdown = max_cum_pnl - cum_pnl
        drawdown_duration = (df_backtest.index[i] - df_backtest.index[(df_backtest.cum_pnl==max_cum_pnl) & (df_backtest.index<df_backtest.index[i])][-1]).days
        
        # entry condition
        enter_pos = False
        exit_code = None
        
        if (not out_pos) and (slope > 0) and allow_enter:
            if z > entry_lvl: 
                res_ratio = get_round_lot_ratio(lot_y=lot_y, lot_x=lot_x, px_y=px_y, px_x=px_x, ratio=slope, max_notional=cum_pnl, value_ratio=log_transform)
                pos_y = -res_ratio.get('pos_y')
                pos_x = res_ratio.get('pos_x')
                enter_pos = True
                out_pos = True
            elif z < -entry_lvl:
                res_ratio = get_round_lot_ratio(lot_y=lot_y, lot_x=lot_x, px_y=px_y, px_x=px_x, ratio=slope, max_notional=cum_pnl, value_ratio=log_transform)
                pos_y = res_ratio.get('pos_y')
                pos_x = -res_ratio.get('pos_x')
                enter_pos = True
                out_pos = True
            
            if enter_pos:
                # rescale position to notional value, assume no leverage
                itrade += 1
                trade_pnl = 0.0
                trade_ratio =  np.abs(pos_x*px_x/(pos_y*px_y) if log_transform else pos_x / pos_y)
                df_trade.loc[itrade, ['entry_time','pos_y','pos_x','entry_px_y','entry_px_x','entry_notional','entry_value','signal_ratio','trade_ratio']] = \
                    [df_backtest.index[i], pos_y, pos_x, px_y, px_x, np.abs(pos_y*px_y) + np.abs(pos_x*px_x), pos_y*px_y + pos_x*px_x, slope, trade_ratio]
        
        # reset stop trade flag
        if (not allow_enter) and (not out_pos) and (slope > 0.0):
            if (stop_pos_y>0 and z>-exit_lvl) or (stop_pos_y<0 and z<exit_lvl):
                allow_enter = True
                stop_pos_y = 0.0
        
        # exit conditions
        if out_pos and (not enter_pos):
            # stop by signal
            if (pos_y>0 and z>-exit_lvl) or (pos_y<0 and z<exit_lvl):
                pos_y = 0.0
                pos_x = 0.0
                exit_code = 'signal'
                
            # stop by drawdown
            if drawdown > stop_drawdown:
                pos_y = 0.0
                pos_x = 0.0
                exit_code = 'drawdown'
            
            # stop by trade pnl
            if trade_pnl < -stop_trade_pnl:
                stop_pos_y = pos_y
                pos_y = 0.0
                pos_x = 0.0
                allow_enter = False
                exit_code = 'trade_pnl'
    
            # reset pnl, cancel outstanding position flag, record trade exit
            if not exit_code == None:
                df_trade.loc[itrade, ['exit_time','exit_code','exit_px_y','exit_px_x']] = [df_backtest.index[i], exit_code, px_y, px_x]
                df_trade.loc[itrade, ['pnl','duration']] = [trade_pnl, (df_trade.loc[itrade, 'exit_time'] - df_trade.loc[itrade, 'entry_time']).days]
                trade_pnl = 0.0
                out_pos = False
    
        # set trade code: enter/exit (pos/neg), long/short (1/2)
        new_trade_code = 0
        if enter_pos:
            new_trade_code = 1 if pos_y > 0 else 2
        elif exit_code != None:
            new_trade_code = -1 if df_backtest['pos_y'].iloc[i-1] > 0 else -2
    
        # record results
        df_backtest.loc[df_backtest.index[i], ['const','slope','sigma','e','z','pos_y','pos_x','trade_ratio','daily_pnl','daily_return','cum_pnl','drawdown','drawdown_duration', 'new_trade_code']] \
            = [const, slope, sigma, e, z, pos_y, pos_x, trade_ratio, daily_pnl, daily_return, cum_pnl, drawdown, drawdown_duration, new_trade_code]

    # calculate aggregate results
    res_summary = {'x': x_name, 
                   'y': y_name,
                   'pair_name': y_name + '-'  + x_name, 
                   'log_transform': log_transform, 
                   'strategy_type': strategy_type, 
                   'ols_lookback_days': ols_lookback_days,
                   'entry_lvl': entry_lvl, 
                   'exit_lvl': exit_lvl, 
                   'stop_trade_pnl': stop_trade_pnl, 
                    'start_notional': df_backtest['cum_pnl'].iloc[0], 
                    'total_pnl': df_backtest['cum_pnl'].iloc[-1] - df_backtest['cum_pnl'].iloc[0], 
                    'max_drawdown': np.max(df_backtest['drawdown']), 
                    'max_drawdown_duration': np.max(df_backtest['drawdown_duration']), 
                    'avg_trade_duration': np.mean(df_trade.duration), 
                    'max_trade_duration': np.max(df_trade.duration), 
                    'num_trade': len(df_trade.index), 
                    'pct_win_trade': np.sum(df_trade.pnl>0) / np.sum(np.isfinite(pd.to_numeric(df_trade.pnl))), 
                    'avg_pnl': np.mean(df_trade.pnl), 
                    'max_profit': np.max(df_trade.pnl[df_trade.pnl>0]), 
                    'max_loss': np.min(df_trade.pnl[df_trade.pnl<0]), 
                    'avg_daily_return': np.mean(df_backtest.daily_return[df_backtest.daily_return!=0.0]), 
                    'std_daily_return': np.std(df_backtest.daily_return[df_backtest.daily_return!=0.0])}
    
    res_summary['sharpe_ratio'] = 16.0 * res_summary.get('avg_daily_return') / res_summary.get('std_daily_return')

    # print results
    if print_results:
        plot_functions.plot_one_series(df_backtest, 'const')
        plot_functions.plot_one_series(df_backtest, 'slope')
        plot_functions.plot_two_series(df_backtest, 'e', 'sigma')
        plot_functions.plot_one_series(df_backtest, 'z', 'new_trade_code')
        plot_functions.plot_one_series(df_backtest, 'cum_pnl', 'new_trade_code')
        plot_functions.plot_one_series(df_backtest, 'daily_pnl')
        plot_functions.plot_density(df_backtest, 'z', 10, True, 'Spread Signal')
        plot_functions.plot_density(df_trade, 'pnl', 10, False, 'Trade PnL')
        
        print('\nPair: %s' % res_summary.get('pair_name'))
        print('Start notional: $%1.0f' % res_summary.get('start_notional'))
        print('Total PnL: $%0.1f' % res_summary.get('total_pnl'))
        print('Total return: %0.1f%%' % (100.0* res_summary.get('total_pnl') / res_summary.get('start_notional')))
        print('\nMax Drawdown: $%0.1f' % res_summary.get('max_drawdown'))
        print('Max Drawdown Duration: %1.0f days' % res_summary.get('max_drawdown_duration'))
        print('Avg Trade Duration: %1.0f days' % res_summary.get('avg_trade_duration'))
        print('Max Trade Duration: %1.0f days' % res_summary.get('max_trade_duration'))
        print('\nAvg daily return: %0.2f%%' % (100.0 * res_summary.get('avg_daily_return')))
        print('Avg annualized return: %0.1f%%' % (100.0 * 256.0 * res_summary.get('avg_daily_return')))
        print('Std daily return: %0.2f%%' % (100.0 * res_summary.get('std_daily_return')))
        print('Std annualized return: %0.1f%%' % (100.0*16.0* res_summary.get('std_daily_return')))
        print('Sharpe ratio: %0.1f' % res_summary.get('sharpe_ratio'))
        
        print('\nNum trades: %1.0f \nWin Pct: %1.0f%% \nAvg pnl: $%0.1f \nMax profit: $%0.1f \nMax loss: $%0.1f' % \
              (res_summary.get('num_trade'), 100.0*res_summary.get('pct_win_trade'), res_summary.get('avg_pnl'), res_summary.get('max_profit'), res_summary.get('max_loss')))
        
    return {'summary':res_summary, 'history':df_backtest, 'trades':df_trade}





if __name__=="__main__":
    
    # test data
    n = 100
    np.random.seed(3)
    w = np.cumsum(np.random.normal(0, 1, n))    
    sigma = 0.99
    high = 15*np.exp(-sigma**2 / 2 + w * sigma / np.sqrt(n))
    low = high * 0.8
    df_data = pd.DataFrame({'low': low, 'high':high}, index = pd.date_range('2018-01-01', periods=100, freq='D'))
    
    
    # test fisher transform
    df_fisher = get_fisher_transform_series(df_data=df_data, ts_high='high', ts_low='low', 
                                            lookback=10, px_weight=0.33, fisher_weight=0.5, return_fisher_only=False)
    pf.plot_two_series(df_fisher, 'px', 'fisher')


    