# Fruit classification project with KNN algorithm

## Thuật toán K-Nearest Neighbor

### Định nghĩa

K – nearest neighbor (KNN) là một trong những thuật toán học có giám sát đơn giản nhất trong Machine Learning. Ý tưởng của KNN là tìm ra output của dữ liệu dựa trên thông tin của những dữ liệu training gần nó nhất.
Kỹ thuật phân lớp dựa trên láng giềng gần nhất được sử dụng rộng rãi trong các hệ thống nhận dạng mẫu, nhận dạng đối tượng, nhận dạng sự kiện, phân loại dữ liệu văn bản… Khái niệm “láng giềng” dùng để chỉ các đối tượng có khoảng cách hoặc độ tương đồng “gần” với đối tượng x. Từ đây, ta cần phải định nghĩa một độ đo khoảng cách hoặc độ đo sự khác biệt giữa các đối tượng.

### Quy trình làm việc của thuật toán KNN

*	Bước 1: Xác định tham số K = số láng giềng gần nhất
*	Bước 2: Tính khoảng cách đối tượng cần phần lớp với tất cả các đối tượng trong training data.
*	Bước 3: Sắp xếp khoảng cách theo thứ tự tăng dần và xác định K láng giềng gần nhất với đối tượng cần phân lớp
*	Bước 4: Lấy tất cả các lớp của K láng giềng gần nhất.
*	Bước 5: Dựa vào phần lớn lớp của K để xác định lớp cho đối tượng cần phân lớp.

### Ví dụ về KNN nhiễu

![image](https://github.com/ThanhTungPh2/Fruit-classification/assets/78742865/57fc412c-301e-4653-b2d3-37c723a45bcf)

Hình trên là bài toán phân lớp với ba lớp: đỏ, lam, lục. Mỗi điểm dữ liệu mới sẽ được gán nhãn theo màu của điểm đó mà nó thuộc về. Trong hình này, chú ý vùng khoanh tròn. Ta nhận thấy rằng điểm màu lục nằm giữa hai vùng lớn với nhiều dữ liệu đỏ và lam, điểm này rất có thể là nhiễu dẫn đến việc dữ liệu test nếu rơi vào vùng này sẽ có nhiều khả năng cho kết quả sai lệch.

### Độ đo khoảng cách

Để xác định độ “gần nhau” giữa hai mẫu dữ liệu, người ta sử dụng một độ đo khoảng cách được định nghĩa trước. Tùy theo kiểu dữ liệu của mẫu và đặc điểm của đối tượng nhận dạng mà ta sử dụng một độ đo phù hợp. Có rất nhiều độ đo khoảng cách (hay độ khác biệt) đã được định nghĩa.
Xét một mẫu dữ liệu x gồm m thuộc tính. Khi đó, mẫu x được xem là một véc tơ trong không gian m chiều (x có m thành phần). Gọi x(x1, x2,…, xm) và y(y1, y2,…, ym) là hai mẫu dữ liệu. Để tính khoảng cách giữa x và y, ký hiệu d(x, y) ta thường sử dụng một số độ đo sau:
Với dữ liệu kiểu số

Khoảng cách Euclidean : 
![Khoảng cách Euclidean](https://github.com/ThanhTungPh2/Fruit-classification/assets/78742865/a73b40e2-372c-4f98-9ca2-bd8336996b02)

## DATASET
|fruit_label	|fruit_name|	fruit_subtype	|mass|	width	|height|	color_score|
|---|---|---|---|---|---|---|
1|apple|granny_smith|192|8.4|7.3|0.55|
|1|apple|granny_smith|180|8|6.8|0.59|
|1|apple|granny_smith|176|7.4|7.2|0.6|
|2|mandarin|mandarin|86|6.2|4.7|0.8|
|2|mandarin|mandarin|84|6|4.6|0.79|
|2|mandarin|mandarin|80|5.8|4.3|0.77|
|2|mandarin|mandarin|80|5.9|4.3|0.81|
|2|mandarin|mandarin|76|5.8|4|0.81|
|1|apple|braeburn|178|7.1|7.8|0.92|
|1|apple|braeburn|172|7.4|7|0.89|
|1|apple|braeburn|166|6.9|7.3|0.93|
|1|apple|braeburn|172|7.1|7.6|0.92|
|1|apple|braeburn|154|7|7.1|0.88|
|1|apple|golden_delicious|164|7.3|7.7|0.7|
|1|apple|golden_delicious|152|7.6|7.3|0.69|
|1|apple|golden_delicious|156|7.7|7.1|0.69|
|1|apple|golden_delicious|156|7.6|7.5|0.67|
|1|apple|golden_delicious|168|7.5|7.6|0.73|
|1|apple|cripps_pink|162|7.5|7.1|0.83|
|1|apple|cripps_pink|162|7.4|7.2|0.85|
|1|apple|cripps_pink|160|7.5|7.5|0.86|
|1|apple|cripps_pink|156|7.4|7.4|0.84|
|1|apple|cripps_pink|140|7.3|7.1|0.87|
|1|apple|cripps_pink|170|7.6|7.9|0.88|
|3|orange|spanish_jumbo|342|9|9.4|0.75|
|3|orange|spanish_jumbo|356|9.2|9.2|0.75|
|3|orange|spanish_jumbo|362|9.6|9.2|0.74|
|3|orange|selected_seconds|204|7.5|9.2|0.77|
|3|orange|selected_seconds|140|6.7|7.1|0.72|
|3|orange|selected_seconds|160|7|7.4|0.81|
|3|orange|selected_seconds|158|7.1|7.5|0.79|
|3|orange|selected_seconds|210|7.8|8|0.82|
|3|orange|selected_seconds|164|7.2|7|0.8|
|3|orange|turkey_navel|190|7.5|8.1|0.74|
|3|orange|turkey_navel|142|7.6|7.8|0.75|
|3|orange|turkey_navel|150|7.1|7.9|0.75|
|3|orange|turkey_navel|160|7.1|7.6|0.76|
|3|orange|turkey_navel|154|7.3|7.3|0.79|
|3|orange|turkey_navel|158|7.2|7.8|0.77|
|3|orange|turkey_navel|144|6.8|7.4|0.75|
|3|orange|turkey_navel|154|7.1|7.5|0.78|
|3|orange|turkey_navel|180|7.6|8.2|0.79|
|3|orange|turkey_navel|154|7.2|7.2|0.82|
|4|lemon|spanish_belsan|194|7.2|10.3|0.7|
|4|lemon|spanish_belsan|200|7.3|10.5|0.72|
|4|lemon|spanish_belsan|186|7.2|9.2|0.72|
|4|lemon|spanish_belsan|216|7.3|10.2|0.71|
|4|lemon|spanish_belsan|196|7.3|9.7|0.72|
|4|lemon|spanish_belsan|174|7.3|10.1|0.72|
|4|lemon|unknown|132|5.8|8.7|0.73|
|4|lemon|unknown|130|6|8.2|0.71|
|4|lemon|unknown|116|6|7.5|0.72|
|4|lemon|unknown|118|5.9|8|0.72|
|4|lemon|unknown|120|6|8.4|0.74|
|4|lemon|unknown|116|6.1|8.5|0.71|
|4|lemon|unknown|116|6.3|7.7|0.72|
|4|lemon|unknown|116|5.9|8.1|0.73|
|4|lemon|unknown|152|6.5|8.5|0.72|
|4|lemon|unknown|118|6.1|8.1|0.7||



