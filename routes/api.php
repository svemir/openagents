<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| is assigned the "api" middleware group. Enjoy building your API!
|
*/

Route::middleware(['auth:sanctum'])->group(function () {
    Route::post('/agents', function (Request $request) {
        return response()->json([
            'name' => $request->name,
            'description' => $request->description,
        ], 201);
    })->name('api.agents.store');

    Route::get('/user', function (Request $request) {
        return $request->user();
    });
});