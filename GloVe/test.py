def common_word(data):
    like = {}
    hate = {}
    for line in data:
        if "like" in line:
            for w in line:
                if w != "like":
                    like[w] = like.get(w, 0) + 1
        elif "hate" in line:
            for w in line:
                if w != "hate":
                    hate[w] = hate.get(w, 0) + 1

    like_most = max(like.items(), key=lambda x: x[1])
    hate_most = max(hate.items(), key=lambda x: x[1])
    return like_most, hate_most