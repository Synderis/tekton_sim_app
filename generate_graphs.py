import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import inflect


def output_graph(results_df):
    p = inflect.engine()
    trials = len(results_df)
    # print(trials)
    tick_times = results_df['tick_times'].tolist()
    no_anvils = len(results_df[(results_df['anvil_count'] == 0)].copy())
    one_anvils = len(results_df[(results_df['anvil_count'] == 1)].copy())
    two_anvils = len(results_df[(results_df['anvil_count'] == 2)].copy())
    three_or_more_anvils = len(results_df[(results_df['anvil_count'] >= 3)].copy())
    no_h_one_a = len(results_df[(results_df['hammer_count'] == 0) & (results_df['anvil_count'] == 1)].copy())
    no_hammer_total = len(results_df[(results_df['hammer_count'] == 0)].copy())
    one_h_one_a = len(results_df[(results_df['hammer_count'] == 1) & (results_df['anvil_count'] == 1)].copy())
    one_hammer_total = len(results_df[(results_df['hammer_count'] == 1)].copy())
    two_h_one_a = len(results_df[(results_df['hammer_count'] == 2) & (results_df['anvil_count'] == 1)].copy())
    two_hammer_total = len(results_df[(results_df['hammer_count'] == 2)][['hammer_count']].copy())
    tick_times_df = results_df.copy()
    tick_times_df['completion'] = 1
    tick_times_df = tick_times_df[['tick_times', 'completion']]
    tick_times_one_anvil = tick_times_df[tick_times_df['tick_times'] <= 150][['tick_times']].copy()

    tick_times_raw = tick_times
    hist2, bin_edges2 = np.histogram(tick_times_raw, density=True)
    diff_bin_edge = np.diff(bin_edges2)
    data_ = hist2 * diff_bin_edge

    cum_cdf_raw = np.cumsum(data_, axis=0)
    sub_115 = len(results_df[(results_df['tick_times'] <= 125)].copy())
    sub_100 = len(results_df[(results_df['tick_times'] <= 100)].copy())

    def output_formatter(numerator, divisor, long):
        if long:
            output = f'{str(numerator)}, {str(round(((numerator / divisor) * 100), 2))}%'
        else:
            output = f'{str(round(((numerator / divisor) * 100), 2))}%'
        return output

    no_anvil_num = output_formatter(no_anvils, trials, True)
    one_anvil_num = output_formatter(one_anvils, trials, True)
    two_anvil_num = output_formatter(two_anvils, trials, True)
    three_anvil_num = output_formatter(three_or_more_anvils, trials, True)
    no_ham_rate_tot = output_formatter(no_h_one_a, trials, False)
    one_ham_rate_tot = output_formatter(one_h_one_a, trials, False)
    two_ham_rate_tot = output_formatter(two_h_one_a, trials, False)

    one_ham_reset = output_formatter((one_h_one_a + two_h_one_a), (one_hammer_total + two_hammer_total), False)
    two_ham_reset = output_formatter(two_h_one_a, two_hammer_total, False)
    no_ham_rate = output_formatter(no_h_one_a, one_anvils, True)
    one_ham_rate = output_formatter(one_h_one_a, one_anvils, True)
    two_ham_rate = output_formatter(two_h_one_a, one_anvils, True)
    sub_115_df = output_formatter(sub_115, one_anvils, True)
    sub_100_df = output_formatter(sub_100, one_anvils, True)

    table_dataframe = pd.DataFrame(
        {('trials = ' + str(trials)): ['no anvil', 'one anvil', 'two anvil', 'three or more'],
         'total, % total': [no_anvil_num, one_anvil_num, two_anvil_num, three_anvil_num],
         'total, sub 1:15 %': ['N/A', sub_115_df, '0', '0'],
         'total, sub 1:00 %': ['N/A', sub_100_df, '0', '0']})
    table_dataframe2 = pd.DataFrame({('trials = ' + str(trials)): ['no hammer', 'one hammer', 'two hammer'],
                                     'total, % of 1 anvils': [no_ham_rate, one_ham_rate, two_ham_rate],
                                     '% of total trials': [no_ham_rate_tot, one_ham_rate_tot, two_ham_rate_tot],
                                     ('1 anvil ham. ' + r'$\subset$' + ' total ham.'): ['N/A', (
                                                 str(one_ham_reset) + ' 1 & 2 Ham.'), (
                                                                                                    str(two_ham_reset) + ' 2 ham. only')]})
    minutes_list = ['0:45', '0:48', '0:51', '0:54', '0:57', '1:00', '1:03', '1:06', '1:09', '1:12', '1:15',
                    '1:18', '1:21', '1:24', '1:27', '1:30', '1:33', '1:36', '1:39', '1:42', '1:45', '1:48',
                    '1:51', '1:54', '1:57', '2:00', '2:03', '2:06', '2:09', '2:12', '2:15', '2:18', '2:21',
                    '2:24', '2:27', '2:30', '2:33', '2:36', '2:39', '2:42', '2:45', '2:48']
    minutes_list_big_step = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15',
                             '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45',
                             '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',
                             '7:30', '7:45', '8:00', '8:15', '8:30', '8:45', '9:00', '9:15', '9:30', '9:45',
                             '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15',
                             '12:30', '12:45', '13:00', '13:15', '13:30', '13:45']
    minutes_list_bigger_step = ['0:00', '0:25', '0:50', '1:15', '1:40', '2:05', '2:30', '2:55', '3:20', '3:45',
                                '4:10', '4:35', '5:00', '5:25', '5:50', '6:15', '6:40', '7:05', '7:30', '7:55',
                                '8:20', '8:45', '9:10', '9:35', '10:00', '10:25', '10:50', '11:15', '11:40', '12:05',
                                '12:30', '12:55', '13:20', '13:45']

    def plot_adjustment():
        plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
        # plt.subplots_adjust(wspace=.2, hspace=0)

    plt.rcParams.update(
        {"lines.color": "silver", "patch.edgecolor": "black", "text.color": "black", "text.antialiased": "True", "axes.facecolor": "silver",
         "axes.edgecolor": "white", "axes.labelcolor": "#ff0000", "xtick.color": "white", "ytick.color": "white",
         "grid.color": "silver", "figure.facecolor": (0.0, 0.0, 0.0, 0.0), "figure.edgecolor": "white",
         "axes.titlecolor": "#ff0000",
         "savefig.facecolor": "black", "savefig.edgecolor": "black"})

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    total_sample_main_plot = fig.add_subplot(gs[2, 1])
    one_anvil_main_plot = fig.add_subplot(gs[0, 1])
    ax4t = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 0])
    ax4t.axes.set_visible(False)
    xd = ['gray', 'gray', 'gray', 'gray']
    silver = ["silver", "silver", "silver", "silver"]
    colors_list = [silver, silver, silver, silver]
    mpl_table = ax1.table(cellText=table_dataframe.values,
                          colLabels=table_dataframe.columns, cellLoc='center', rowLoc='center', loc='upper right',
                          cellColours=colors_list, colColours=xd)
    mpl_table.auto_set_font_size(False)
    ax1.axis(False)
    mpl_table.set_fontsize(9)

    mpl_table2 = ax2.table(cellText=table_dataframe2.values,
                           colLabels=table_dataframe2.columns, cellLoc='center', rowLoc='center', loc='upper right',
                           cellColours=colors_list[0:3], colColours=xd)
    mpl_table2.auto_set_font_size(False)
    mpl_table2.set_fontsize(9)
    ax2.axis(False)

    n = pd.Series(np.random.randn(trials))
    first_q1 = float(np.quantile(n, .25))
    first_q3 = float(np.quantile(n, .75))
    iqr = first_q3 - first_q1
    bin_width = (2 * iqr) / ((len(n)) ** (1. / 3.))
    bin_number = int(np.ceil((np.max(n) - np.min(n)) / bin_width))
    m = pd.Series(np.random.randn(len(tick_times_one_anvil)))
    second_q1 = float(np.quantile(m, .25))
    second_q3 = float(np.quantile(m, .75))
    iqr_second = second_q3 - second_q1
    bin_width_second = (2 * iqr_second) / ((len(m)) ** (1. / 3.))
    bin_number_second = int(np.ceil((m.max() - m.min()) / bin_width_second))

    cumulative_total_graph = sns.kdeplot(tick_times_df, x='tick_times', cumulative=True, common_norm=False,
                                         common_grid=True, legend=True, color='crimson', linewidth=2)
    empirical_total_graph = sns.ecdfplot(tick_times_df, x='tick_times', legend=True, color='k', linewidth=1.75)
    data_x, data_y = cumulative_total_graph.lines[0].get_data()
    yi = .99
    xi = np.interp(yi, data_y, data_x)
    cumulative_total_graph.set(xticks=(np.arange(0, 1400, step=25)), xlim=(0, xi), yticks=(np.arange(0, 1.1, step=.1)),
                               ylim=(0, 1), ylabel='probability of killing tekton', xlabel='time of encounter in ticks',
                               title='cumulative probability of killing tekton')
    cumulative_total_graph.set_xticklabels(cumulative_total_graph.get_xticklabels(), rotation=45)
    aux_axis_cumulative = cumulative_total_graph.twiny()
    sns.kdeplot(ax=aux_axis_cumulative, bins=80)
    cumulative_total_graph.legend(labels=('theoretical', 'empirical'), labelcolor='black')
    aux_axis_cumulative.set(xticks=(np.arange(0, 840, step=25)), xlim=(0, (xi * .6)),
                            xlabel='time of encounter in seconds')
    aux_axis_cumulative.set_xticklabels(minutes_list_bigger_step, rotation=45)
    cumulative_total_graph.grid('visible', color='black')

    # total histogram graph
    with sns.axes_style(style='ticks', rc={'ytick.left': True}):
        sns.histplot(tick_times, bins=bin_number, ax=total_sample_main_plot, color='orange', alpha=1, edgecolor='k', linewidth=1)
    total_sample_main_plot_aux_axis = total_sample_main_plot.twinx()

    sns.kdeplot(tick_times, ax=total_sample_main_plot_aux_axis, color='crimson')

    total_sample_main_plot_xticks = (np.arange(75, (np.max(tick_times) + 25), step=25))
    total_sample_main_plot.xaxis.set_tick_params(rotation=45)
    total_sample_main_plot_aux_axis.set(ylabel='Probability density')
    total_sample_main_plot.set(ylabel='number of killed tektons in sample')
    total_sample_main_plot_aux_xaxis = total_sample_main_plot.secondary_xaxis('top')
    total_sample_main_plot_aux_xaxis.xaxis.set_tick_params(rotation=45)
    total_sample_main_plot_aux_xaxis.set(xticks=(np.arange(75, (np.max(tick_times)), step=25)),
                                         xlim=(75, (np.max(tick_times))))
    total_sample_main_plot_aux_xaxis_labels = np.arange(75, (np.max(tick_times)), step=25)
    total_sample_main_plot_aux_xaxis.set_xticklabels(
        minutes_list_big_step[3:(len(total_sample_main_plot_aux_xaxis_labels) + 3)])
    total_sample_main_plot.set(title='tekton density histogram of ' + p.number_to_words(trials) + ' trials',
                               xticks=total_sample_main_plot_xticks, xlim=(75, (np.max(tick_times))))
    total_sample_main_plot.locator_params(nbins=22, axis='y')
    total_sample_main_plot_aux_axis.locator_params(nbins=22, axis='y')
    total_sample_main_plot.set_xlabel('time of encounter in ticks')
    total_sample_main_plot_aux_xaxis.set_xlabel('time of encounter in minutes:seconds')
    total_sample_main_plot.xaxis.grid(True, color='black')
    total_sample_main_plot.yaxis.grid(True, color='black')
    total_sample_main_plot.set_axisbelow(True)

    # under one anvil graph
    with sns.axes_style(style='ticks', rc={'ytick.left': True}):
        sns.histplot(tick_times_one_anvil, bins=bin_number_second, ax=one_anvil_main_plot, palette=['orange'], alpha=1,
                     legend=False, edgecolor='k', linewidth=1)

    one_anvil_main_plot_aux_axis = one_anvil_main_plot.twinx()
    sns.kdeplot(tick_times_one_anvil, palette=['r'], ax=one_anvil_main_plot_aux_axis, legend=False)

    one_anvil_main_plot.xaxis.set_tick_params(rotation=45)
    one_anvil_main_plot_aux_axis.set(ylabel='Probability density')
    one_anvil_main_plot.set(ylabel='number of killed tektons in sample')
    one_anvil_main_plot_aux_xaxis = one_anvil_main_plot.secondary_xaxis('top')
    one_anvil_main_plot_aux_xaxis.xaxis.set_tick_params(rotation=45)

    one_anvil_main_plot_xticks = (np.arange(75, 165, step=5))
    one_anvil_main_plot.set(xticks=one_anvil_main_plot_xticks, xlim=(75, 160),
                            title='number of tektons under one anvil in ' + p.number_to_words(trials) + ' trials')
    one_anvil_main_plot.locator_params(nbins=22, axis='y')
    one_anvil_main_plot.set_xlabel('time of encounter in ticks')
    one_anvil_main_plot_aux_xaxis.set_xlabel('time of encounter in minutes:seconds')
    one_anvil_main_plot_aux_axis.locator_params(nbins=22, axis='y')
    one_anvil_main_plot.xaxis.set_tick_params(rotation=45)

    one_anvil_main_plot_aux_xaxis.set(xticks=(np.arange(75, 165, step=5)), xlim=(75, 160))
    one_anvil_main_plot_aux_xaxis.set_xticklabels(minutes_list[:18])
    one_anvil_main_plot_aux_xaxis.xaxis.set_tick_params(rotation=45)
    one_anvil_main_plot.xaxis.grid(True, color='black')
    one_anvil_main_plot.yaxis.grid(True, color='black')
    one_anvil_main_plot.set_axisbelow(True)

    plt.subplots_adjust(wspace=.25, hspace=0, right=.93, left=0.05, top=.90, bottom=.07)
    return fig
