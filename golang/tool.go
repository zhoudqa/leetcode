package golang

func makeMatrixWithInitialFunc[T comparable](length, width int, initFunc func(i, j int) T) [][]T {
	cache := make([][]T, length)
	for i := 0; i < length; i++ {
		cache[i] = make([]T, width)
		if initFunc != nil {
			for j := range cache[i] {
				cache[i][j] = initFunc(i, j)
			}
		}
	}
	return cache
}

func makeCacheWithInitialFunc[T comparable](length int, initFunc func(i int) T) []T {
	cache := make([]T, length)
	if initFunc != nil {
		for i := 0; i < length; i++ {
			cache[i] = initFunc(i)
		}
	}
	return cache
}
