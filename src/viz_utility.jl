import PyPlot
using PyCall
@pyimport wordcloud

function visualize_bartopics(topics::Matrix{Float64})
    gray_cmp = PyPlot.ColorMap("gray")
    KK, vv = size(topics)
    dd = convert(Int64, sqrt(vv))

    fig, axes = PyPlot.plt[:subplots](1, KK)
    axes = collect(axes)

    for kk = 1:KK
        axes[kk][:imshow](reshape(topics[kk, :], dd, dd), cmap=gray_cmp)
        axes[kk][:axis]("off")
    end
        return fig
end

function plot_log_likelihood(log_likelihood)
    fig, ax = PyPlot.subplots()
    ax[:plot](1:length(log_likelihood), log_likelihood)
    return fig
end


function topic2png(filename, vocab, alpha)

  idx_sorted = sortperm(alpha, rev=true)

  words = Array(Tuple{ASCIIString, Float64}, length(vocab))
  for i = 1:length(idx_sorted)
    words[i] = (vocab[idx_sorted[i]], alpha[idx_sorted[i]])
  end


  wc = wordcloud.WordCloud(background_color="white", max_font_size=50, relative_scaling=0.5)
  wc[:fit_words](words)

  fig, ax = PyPlot.subplots()
  ax[:imshow](wc);
  ax[:axis]("off")
  fig[:savefig](filename, format="png")
end